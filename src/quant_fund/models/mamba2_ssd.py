"""Mamba-2 / SSD (Dao & Gu 2024) — scalar-per-channel selective SSM (SYNTHETIC)
h_t = a_t h_{t-1} + b_t x_t with b_t,c_t,a_t input-dependent (the SSD
restricted form = attention-like dual), trained on induction recall.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mamba2_ssd requires torch (pip install -e .[nn])") from exc
    return torch


def bench_mamba2_ssd(seed: int = 2243, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pa = torch.nn.Linear(D, D)  # a_t logit
    pb = torch.nn.Linear(D, D)  # b_t
    pc = torch.nn.Linear(D, D)  # c_t
    head = torch.nn.Linear(D, VOCAB)
    params = (
        list(emb.parameters())
        + list(pa.parameters())
        + list(pb.parameters())
        + list(pc.parameters())
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        a = torch.sigmoid(pa(h))  # scalar-per-channel decay
        b = torch.sigmoid(pb(h))
        c = pc(h)
        s = torch.zeros(B_, D)
        for t in range(T_):
            s = a[:, t] * s + b[:, t] * h[:, t]
        return (c[:, -1] * s).sum(-1, keepdim=True).expand(B_, D)

    for _ in range(iters):
        o = scan(emb(X))
        loss = torch.nn.functional.cross_entropy(head(o), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_mamba2_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_mamba2_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
