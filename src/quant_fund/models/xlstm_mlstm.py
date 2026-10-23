"""xLSTM / mLSTM (Beck et al. 2024) — matrix-memory LSTM: (SYNTHETIC)
M_t = f_t M_{t-1} + i_t k_t v_t^T, readout (q_t^T M)/(q_t^T Σ f k).
Covariance-matrix memory with learned gates on induction recall.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("xlstm_mlstm requires torch (pip install -e .[nn])") from exc
    return torch


def bench_xlstm_mlstm(seed: int = 2251, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pq = torch.nn.Linear(D, D)
    pk = torch.nn.Linear(D, D)
    pv = torch.nn.Linear(D, D)
    pf = torch.nn.Linear(D, 1)
    pi = torch.nn.Linear(D, 1)
    head = torch.nn.Linear(D, VOCAB)
    params = (
        list(emb.parameters())
        + list(pq.parameters())
        + list(pk.parameters())
        + list(pv.parameters())
        + list(pf.parameters())
        + list(pi.parameters())
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        M = torch.zeros(B_, D, D)
        normalizer = torch.zeros(B_, D)
        for t in range(T_):
            f = torch.sigmoid(pf(h[:, t]))
            i = torch.sigmoid(pi(h[:, t]))
            k = torch.tanh(pk(h[:, t]))
            v = pv(h[:, t])
            M = f[:, :, None] * M + i[:, :, None] * torch.einsum("bi,bj->bij", k, v)
            normalizer = f * normalizer + i * k
        q = pq(h[:, -1])
        num = torch.einsum("bi,bij->bj", q, M)
        den = (q * normalizer).sum(-1, keepdim=True).abs().clamp_min(1e-3)
        return num / den

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(head(scan(emb(X))), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_mlstm_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_mlstm_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
