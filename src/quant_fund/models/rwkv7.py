"""RWKV-7 (2025) — generalized delta rule with vector-valued decay: (SYNTHETIC)
S_t = S_{t-1}(Diag(w_t) - beta_t k_t (k_t^T Diag(w_t))) + beta_t k_t v_t^T;
state tracks and corrects itself via the delta rule. Induction recall.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("rwkv7 requires torch (pip install -e .[nn])") from exc
    return torch


def bench_rwkv7(seed: int = 2257, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pw = torch.nn.Linear(D, D)
    pk = torch.nn.Linear(D, D)
    pv = torch.nn.Linear(D, D)
    pq = torch.nn.Linear(D, D)
    pb = torch.nn.Linear(D, D)
    head = torch.nn.Linear(D, VOCAB)
    params = (
        list(emb.parameters())
        + list(pw.parameters())
        + list(pk.parameters())
        + list(pv.parameters())
        + list(pq.parameters())
        + list(pb.parameters())
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        S = torch.zeros(B_, D, D)
        for t in range(T_):
            w = torch.sigmoid(pw(h[:, t]))  # vector decay
            k = torch.tanh(pk(h[:, t]))
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            v = pv(h[:, t])
            beta = torch.sigmoid(pb(h[:, t]))
            kw = torch.einsum("bi,bi->bi", k, w)
            decay = torch.diag_embed(w) - beta[:, :, None] * torch.einsum("bi,bj->bij", k, kw)
            S = torch.einsum("bij,bjk->bik", S, decay) + beta[:, :, None] * torch.einsum(
                "bi,bj->bij", k, v
            )
        q = torch.tanh(pq(h[:, -1]))
        return torch.einsum("bi,bij->bj", q, S)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(head(scan(emb(X))), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_rwkv7_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_rwkv7_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
