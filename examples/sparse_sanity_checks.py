"""
Sanity checks using FoldedLayer (FP4 + delta-CSR sparse) with sili autograd.

FoldedLayer has a proper backward pass through the sparse weights, so task
gradients reach the sparse connections. This is what enables actual learning.

Architecture:
  - FoldedLayer (sparse, FP4 delta-CSR) for the hidden projection
  - EnergyDynamics on the hidden activations
  - DenseHead for output (no energy, plain numpy SGD)
  - Task gradient: head → energy gate → FoldedLayer backward → sparse W

The sparse layer uses FP4BiPacked (4-bit weight + 4-bit importance per
connection) with delta-CSR encoding (1-2 bytes per index for typical layers).
Activity-correlation importance updates in forward_dense() track which connections
are actively used, guiding synaptogenesis pruning/growing decisions.
Weight values are changed only through backward_dense() (backprop), not forward.

Run: python -m examples.sparse_sanity_checks --task rare --steps 3000
"""

import argparse, math, zlib
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sili.cpu
from sili.tensor import Tensor
from sili.sparse_rnn import FoldedLayer
from sili.conversion.rnn_fold import FoldedBlockDescriptor
from sili.energy import EnergyDynamics
import torch


# ── Datasets ──────────────────────────────────────────────────────────────────

def gen_rare(T=150):
    x=np.zeros((T,2),np.float32); x[:,0]=1.
    k=np.random.randint(0,max(1,T//2+1)); cur,gap=k,k//2; indices=[k]
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

GENS=dict(rare=gen_rare,copy=gen_copy,char=gen_char)


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat=np.concatenate(bufs).tobytes()
    return len(zlib.compress(flat,1))/max(len(flat),1)


def make_sparse_folded_layer(n_in, n_out, density, lr, seed=None):
    """
    Build a FoldedLayer (n_in -> n_out) with ~density fraction of connections.
    Uses n_folds=1 so it acts as a plain sparse linear layer.
    Task gradients propagate through backward() since FoldedLayer is in the
    sili autograd graph.
    """
    if seed is not None: torch.manual_seed(seed)
    # Random dense matrix, keep only `density` fraction of weights
    W = torch.randn(n_out, n_in) * np.sqrt(2./n_in)  # (n_out, n_in) for torch CSR
    mask = torch.rand_like(W) < density
    W = W * mask
    stacked = W.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()

    desc = FoldedBlockDescriptor(
        n_folds=1,
        block_indices=[0],
        stacked_weights={'.proj.weight': stacked},
        out_dims={'.proj.weight': n_out},
        band_half_widths={'.proj.weight': None},
        prefix='sparse.',
    )
    return FoldedLayer.from_descriptor(desc, learning_rate=lr, num_cpus=1)


class DenseHead:
    """Output head with no energy -- task gradient propagates back to h."""
    def __init__(self, in_f, out_f, lr):
        self.w=np.random.randn(in_f,out_f).astype(np.float32)*np.sqrt(2./in_f)
        self.b=np.zeros(out_f,np.float32); self.lr=lr; self._h=None

    def forward(self, h_np): self._h=h_np.copy(); return h_np@self.w+self.b

    def backward(self, g_out):
        self.w-=self.lr*np.outer(self._h,g_out)
        self.b-=self.lr*g_out
        return g_out@self.w.T  # gradient back to h


def run(task='rare', steps=3000, hidden=32, lr=0.005,
        density=0.25, aux_weight=0.05, report_every=300):
    T=150; n_in=2; n_out=2
    n_all = n_in + hidden  # concatenated input+state

    # FoldedLayer: n_all -> hidden (sparse, FP4, with autograd)
    layer = make_sparse_folded_layer(n_all, hidden, density, lr, seed=0)

    # Energy on hidden activations only
    frac=max(0.1,min(0.4,6./hidden))
    energy=EnergyDynamics(drive=1./hidden,activation_cost=0.05,
        precision=0.01,density=frac/2,exploration=0.001,p=frac)
    k_active=max(1,round(frac*hidden))

    head=DenseHead(hidden,n_out,lr)

    print(f"\n=== SPARSE (FoldedLayer) task={task} hidden={hidden} "
          f"density={density:.0%} nnz={layer.nnz_total()} k_active={k_active} ===")
    print(f"energy_start={energy._energy_start:.2f}")
    print(f"Task gradient: head -> energy gate -> FoldedLayer.backward() -> sparse W")

    correct=total=rare_hit=rare_miss=0; hid_buf=[]

    for step in range(steps):
        seq=GENS[task](T); state=np.zeros(hidden,np.float32)
        for t in range(T):
            # -- sparse forward (sili autograd) --
            inp = Tensor(np.concatenate([seq[t],state]).astype(np.float32))
            h_raw = layer(inp)         # FoldedLayer: Tensor in autograd graph

            # -- energy gating (hidden only) --
            h_out, aux, _ = energy.forward(h_raw)

            # -- output head (no energy) --
            pred = head.forward(h_out.data)
            # Current-step target: proves learning/gradients work.
            # Real temporal tasks would predict next-step (see sanity_checks.py sine task).
            tgt  = seq[t, :n_out]

            # -- task loss gradient --
            lg=pred-pred.max(); prb=np.exp(lg)/np.exp(lg).sum()
            ti=int(tgt.argmax()); g_out=prb.copy(); g_out[ti]-=1.

            # -- gradient chain: head -> energy -> FoldedLayer --
            g_h = head.backward(g_out)   # updates head, returns grad to h
            h_out.grad = g_h             # feed task grad back to energy output
            h_out.backward()             # propagate through energy gate to W

            # -- aux_loss: energy homeostasis (additional gradient path) --
            if aux is not None:
                aux.grad=np.array([aux_weight],np.float32)
                aux.backward()

            state=h_out.data.copy()

            pred_cls=int(pred.argmax()); tgt_cls=int(tgt.argmax())
            correct+=int(pred_cls==tgt_cls); total+=1
            if task=='rare' and tgt_cls==1:
                if pred_cls==1: rare_hit+=1
                else: rare_miss+=1

            hid_buf.append(state.copy())
            if len(hid_buf)>200: hid_buf.pop(0)

        if (step+1)%report_every==0:
            acc=correct/max(1,total); cr=compression_ratio(hid_buf)
            extra=''
            if task=='rare':
                prec=rare_hit/max(1,rare_hit+rare_miss)
                extra=f'  rare_prec={prec:.3f}'
            print(f"  step {step+1:5d}  acc={acc:.3f}{extra}  "
                  f"compress={cr:.3f}  nnz={layer.nnz_total()}")
            correct=total=rare_hit=rare_miss=0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",default="rare",choices=list(GENS))
    ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--hidden",type=int,default=32)
    ap.add_argument("--lr",type=float,default=0.005)
    ap.add_argument("--density",type=float,default=0.25)
    ap.add_argument("--report-every",type=int,default=300)
    a=ap.parse_args()
    run(a.task,a.steps,a.hidden,a.lr,a.density,report_every=a.report_every)

if __name__=="__main__":
    main()
