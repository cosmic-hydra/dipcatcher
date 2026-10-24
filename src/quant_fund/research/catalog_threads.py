"""Pin torch intra-op threads before the research catalog runs.

CI already pins OpenMP, MKL, and OpenBLAS. PyTorch keeps its own pool at
the host CPU count, so one ``run_research`` oversubscribes every small
tensor op. Intra-op threads lock after the first parallel region; set
them before any catalog bench runs.
"""

from __future__ import annotations


def pin_catalog_threads() -> None:
    """One intra-op thread. A checkout without torch is a no-op."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
