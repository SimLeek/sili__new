"""
Sanity checks for sili EnergyDynamics + sili autograd.

Ports EnergyWrapper/example/train_example.py tasks:
  sine -- predict next sin value from [cos, sin] input
  rare -- detect geometrically-spaced rare events
  copy -- repeat 15-bit sub-sequence
  char -- predict even/odd of random digits

Architecture (no torch in compute path):
  - sili Tensor autograd for linear layers
  - EnergyDynamics as the activation/nonlinearity (no separate tanh)
  - aux_loss + task_loss combined to handle fired neurons whose gradient
    path is blocked by the constant-2.0 output (fired = threshold event)
  - Gradient clipping prevents RNN explosion

Learning metric: zlib compression ratio of recent hidden states.
Lower = more varied = better learned representation.

Energy starts at 2.0 - drive*10 so neurons respond within ~10 steps
rather than waiting for buildup from 0.

Run: python -m examples.sanity_checks [--task sine] [--steps 2000]
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
    phi = np.random.uniform(0, 2*math.pi)
    t   = np.arange(T) * 0.1
    return np.stack([np.cos(t+phi), np.sin(t+phi)], 1).astype(np.float32)

def gen_rare(T=150):
    x = np.zeros((T,2), np.float32); x[:,0]=1.
    k = np.random.randint(0, max(1,T//2+1))
    cur, gap = k, k//2
    indices=[k]
    while cur+gap+1 < T and gap > 0: cur+=gap+1; indices.append(cur); gap//=2
    for i in indices: x[i]=[0.,1.]
    return x

def gen_copy(T=150):
    sub=np.random.randint(0,2,15)
    x=np.zeros((T,2),np.float32)
    for t in range(T): x[t,sub[t%15]]=1.
    return x

def gen_char(T=150):
    d=np.random.randint(0,10,T)
    x=np.zeros((T,2),np.float32)
    for t in range(T): x[t,d[t]%2]=1.
    return x

GENS=dict(sine=gen_sine,rare=gen_rare,copy=gen_copy,char=gen_char)


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat=np.concatenate(bufs).tobytes()
    return len(zlib.compress(flat,1))/max(len(flat),1)


# ── Training ──────────────────────────────────────────────────────────────────

def run(task='sine', steps=2000, hidden=64, lr=0.002,
        aux_weight=0.05, clip=1.0, report_every=200):
    T=150; n_in=2
    n_out = 1 if task=='sine' else 2

    # Weights as sili Tensors (in autograd graph)
    scale = 0.05
    W  = Tensor(np.random.randn(n_in+hidden, hidden).astype(np.float32)*scale)
    bW = Tensor(np.zeros(hidden, np.float32))
    V  = Tensor(np.random.randn(hidden, n_out).astype(np.float32)*scale)
    bV = Tensor(np.zeros(n_out, np.float32))
    params=[W,bW,V,bV]

    # Energy: fires when energy >= 2.0; starts pre-charged at energy_start
    # p: fraction of neurons kept active per step. Default p=0.02 gives
    # k=max(1, round(0.02*hidden))=1 active neuron for hidden=64 -- too sparse.
    # Use p=0.2 (20%) for a reasonable capacity on small test models.
    frac_active = max(0.05, min(0.5, 4. / hidden))  # ~4 neurons minimum, max 50%
    energy=EnergyDynamics(
        drive=1./hidden, activation_cost=0.05,
        precision=0.01, density=frac_active/2, exploration=0.001, p=frac_active)
    print(f"\n=== task={task} hidden={hidden} lr={lr} "
          f"energy_start={energy._energy_start:.2f} ===")

    correct=wrong=0
    hid_buf=[]

    for step in range(steps):
        seq=GENS[task](T)
        state=np.zeros(hidden, np.float32)

        for t in range(T-1):
            # -- forward --
            xh    = Tensor(np.concatenate([seq[t], state]))
            h_raw = xh @ W + bW               # in autograd graph
            h_out, aux, _ = energy.forward(h_raw)
            pred  = h_out @ V + bV

            tgt = seq[t+1, :n_out]

            # -- task loss gradient (MSE or CE) --
            if task=='sine':
                err=pred.data-tgt
                pred.grad = (2./n_out)*err
            else:
                lg=pred.data-pred.data.max()
                prb=np.exp(lg)/np.exp(lg).sum()
                ti=int(tgt.argmax())
                pred.grad=prb.copy(); pred.grad[ti]-=1.

            pred.backward()

            # -- aux_loss (energy homeostasis) --
            # Weight by aux_weight to blend with task signal.
            # This reaches fired neurons whose h_out = 2.0 (const),
            # providing gradient even when gate=0 blocks the task path.
            if aux is not None:
                aux.grad = np.array([aux_weight], np.float32)
                aux.backward()

            # -- gradient clip + SGD step --
            for p in params:
                if p.grad is not None:
                    np.clip(p.grad, -clip, clip, out=p.grad)
                    p.data -= lr * p.grad
                    p.grad = None

            # state = energy-gated hidden (energy provides the nonlinearity)
            state = h_out.data.copy()

            # accuracy
            if task=='sine':
                correct += float(abs(pred.data[0]-tgt[0]) < 0.1)
                wrong   += float(abs(pred.data[0]-tgt[0]) >= 0.1)
            else:
                correct += int(pred.data.argmax()==int(tgt.argmax()))
                wrong   += 1-int(pred.data.argmax()==int(tgt.argmax()))

            hid_buf.append(state.copy())
            if len(hid_buf)>200: hid_buf.pop(0)

        if (step+1)%report_every==0:
            tot=max(1,correct+wrong)
            print(f"  step {step+1:5d}  acc={correct/tot:.3f}  "
                  f"compress={compression_ratio(hid_buf):.3f}")
            correct=wrong=0

    print("Done.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",default="sine",choices=list(GENS))
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--lr",type=float,default=0.002)
    ap.add_argument("--aux-weight",type=float,default=0.05)
    ap.add_argument("--report-every",type=int,default=200)
    a=ap.parse_args()
    run(a.task,a.steps,a.hidden,a.lr,a.aux_weight,report_every=a.report_every)

if __name__=="__main__":
    main()
