"""
Sanity checks for sili EnergyDynamics + sili autograd.

Key design decisions:
  - Energy applies to HIDDEN neurons only, not outputs.
    Energy fires neurons to 2.0 regardless of target -- applying it to outputs
    on a dense reward task conflicts with every gradient signal.
  - Task gradient propagates THROUGH the energy layer back to W via sili autograd.
    The DenseHead computes g_h = g_out @ w.T and feeds it back as h_out.grad
    so the task loss reaches the sili Tensor weights.
  - aux_loss (energy homeostasis) provides an additional gradient path through
    fired neurons whose gate blocks the task gradient.

Tasks:
  rare  -- sparse events: best fit for energy RL curiosity
  copy  -- repeated 15-bit sub-sequence: binary, temporally correlated
  char  -- even/odd digit: binary, ~equal classes, moderate density
  sine  -- dense regression: worst fit for energy, baseline only

Init modes:
  normal -- random weights
  zero   -- all zeros. Energy pre-set to 1.9 to immediately escape dead zone.

Run: python -m examples.sanity_checks --task rare --steps 2000
"""

import argparse, math, zlib
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sili.cpu
from sili.tensor import Tensor
from sili.energy import EnergyDynamics


# ── Datasets ──────────────────────────────────────────────────────────────────

def gen_sine(T=150):
    phi=np.random.uniform(0,2*math.pi); t=np.arange(T)*0.1
    return np.stack([np.cos(t+phi),np.sin(t+phi)],1).astype(np.float32)

def gen_rare(T=150):
    x=np.zeros((T,2),np.float32); x[:,0]=1.
    k=np.random.randint(0,max(1,T//2+1))
    cur,gap=k,k//2; indices=[k]
    while cur+gap+1<T and gap>0: cur+=gap+1; indices.append(cur); gap//=2
    for i in indices: x[i]=[0.,1.]
    return x

def gen_copy(T=150):
    sub=np.random.randint(0,2,15); x=np.zeros((T,2),np.float32)
    for t in range(T): x[t,sub[t%15]]=1.
    return x

def gen_char(T=150):
    d=np.random.randint(0,10,T); x=np.zeros((T,2),np.float32)
    for t in range(T): x[t,d[t]%2]=1.
    return x

GENS=dict(sine=gen_sine,rare=gen_rare,copy=gen_copy,char=gen_char)


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat=np.concatenate(bufs).tobytes()
    return len(zlib.compress(flat,1))/max(len(flat),1)


class DenseHead:
    """
    Output head with no energy.
    backward() returns g_h = g_out @ w.T so the task gradient can propagate
    back through the energy layer into the sili Tensor hidden weights.
    """
    def __init__(self, in_f, out_f, lr):
        s=np.sqrt(2./in_f)
        self.w=np.random.randn(in_f,out_f).astype(np.float32)*s
        self.b=np.zeros(out_f,np.float32)
        self.lr=lr; self._h=None

    def zero_init(self): self.w[:]=0.; self.b[:]=0.

    def forward(self, h_np):
        self._h=h_np.copy(); return h_np@self.w+self.b

    def backward(self, g_out):
        """Returns gradient to h so it can propagate to sili Tensor weights."""
        self.w-=self.lr*np.outer(self._h,g_out)
        self.b-=self.lr*g_out
        return g_out@self.w.T  # gradient back to h


# ── Training ──────────────────────────────────────────────────────────────────

def run(task='rare', steps=2000, hidden=32, lr=0.01,
        aux_weight=0.05, clip=1.0, init='normal', report_every=200):
    T=150; n_in=2
    n_out=1 if task=='sine' else 2

    # k_active: ~4-6 neurons minimum for usable gradient coverage
    frac=max(0.1,min(0.4,6./hidden))
    energy=EnergyDynamics(drive=1./hidden,activation_cost=0.05,
        precision=0.01,density=frac/2,exploration=0.001,p=frac)
    k_active=max(1,round(frac*hidden))
    print(f"\n=== task={task}  init={init}  hidden={hidden}  k_active={k_active} ===")
    print(f"energy_start={energy._energy_start:.2f}  (threshold 2.0)")
    print(f"NOTE: energy on hidden only; task gradient propagates back to W via autograd")

    if init=='zero':
        W =Tensor(np.zeros((n_in+hidden,hidden),np.float32))
        bW=Tensor(np.zeros(hidden,np.float32))
        energy.energy=np.full(hidden,1.9,np.float32)  # pre-charged for fast escape
    else:
        scale=np.sqrt(2./(n_in+hidden))
        W =Tensor(np.random.randn(n_in+hidden,hidden).astype(np.float32)*scale)
        bW=Tensor(np.zeros(hidden,np.float32))

    head=DenseHead(hidden,n_out,lr)
    if init=='zero': head.zero_init()
    params=[W,bW]

    correct=total=0; hid_buf=[]
    # For rare task: track precision on class-1 events specifically
    rare_hit=rare_miss=0

    for step in range(steps):
        seq=GENS[task](T); state=np.zeros(hidden,np.float32)
        for t in range(T if task!="sine" else T-1):
            xh=Tensor(np.concatenate([seq[t],state]))
            h_raw=xh@W+bW                          # sili autograd
            h_out,aux,_=energy.forward(h_raw)       # energy gates hidden only

            # Plain dense output -- no energy on predictions
            pred=head.forward(h_out.data)
            # EnergyWrapper pattern: predict CURRENT step's class.
            # For sine, next-step makes sense (temporal prediction).
            # For classification tasks (char/copy/rare), predict current input.
            tgt=seq[t+1,:n_out] if task=='sine' else seq[t,:n_out]

            # Task loss gradient
            if task=='sine':
                err=pred-tgt; g_out=(2./n_out)*err
            else:
                lg=pred-pred.max(); prb=np.exp(lg)/np.exp(lg).sum()
                ti=int(tgt.argmax()); g_out=prb.copy(); g_out[ti]-=1.

            # -- Gradient chain: head → energy gate → sili hidden weights W --
            g_h=head.backward(g_out)   # head update + returns grad to h
            # Feed task gradient back through the energy layer to W and bW
            h_out.grad=g_h             # set grad on the sili Tensor
            h_out.backward()           # propagate to W and bW

            # Energy aux_loss: homeostasis gradient (reaches fired neurons
            # whose gate blocks the task gradient above)
            if aux is not None:
                aux.grad=np.array([aux_weight],np.float32)
                aux.backward()

            for p in params:
                if p.grad is not None:
                    np.clip(p.grad,-clip,clip,out=p.grad)
                    p.data-=lr*p.grad; p.grad=None

            state=h_out.data.copy()

            # Accuracy (correct denominator)
            if task=='sine':
                correct+=float(abs(pred[0]-tgt[0])<0.1); total+=1
            else:
                pred_cls=int(pred.argmax()); tgt_cls=int(tgt.argmax())
                correct+=int(pred_cls==tgt_cls); total+=1
                if task=='rare' and tgt_cls==1:
                    if pred_cls==1: rare_hit+=1
                    else: rare_miss+=1

            hid_buf.append(state.copy())
            if len(hid_buf)>200: hid_buf.pop(0)

        if (step+1)%report_every==0:
            acc=correct/max(1,total)
            cr=compression_ratio(hid_buf)
            extra=''
            if task=='rare':
                prec=rare_hit/max(1,rare_hit+rare_miss)
                extra=f'  rare_precision={prec:.3f}'
            print(f"  step {step+1:5d}  acc={acc:.3f}{extra}  compress={cr:.3f}")
            correct=total=rare_hit=rare_miss=0

    print("Done.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",default="rare",choices=list(GENS))
    ap.add_argument("--init",default="normal",choices=["normal","zero"])
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--hidden",type=int,default=32)
    ap.add_argument("--lr",type=float,default=0.01)
    ap.add_argument("--aux-weight",type=float,default=0.05)
    ap.add_argument("--report-every",type=int,default=200)
    a=ap.parse_args()
    run(a.task,a.steps,a.hidden,a.lr,a.aux_weight,init=a.init,report_every=a.report_every)

if __name__=="__main__":
    main()
