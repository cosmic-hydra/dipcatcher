"""Feature attribution against a proper score.

Primary path: seeded permutation importance — the same estimator protocol as
``sklearn.inspection.permutation_importance`` (baseline score minus mean of
permuted scores over ``n_repeats`` shuffles), implemented directly so any
ForecastModel ``predict`` callable works without an estimator adapter and the
permutation stream is transparently deterministic under ``seed``.

Importance is reported as the *increase* in the proper loss when the feature
column is permuted: positive means the feature helps. Values are never
clipped — a negative mean is reported honestly (permuted column can beat the
baseline by luck on tiny eval sets).

Optional path: SHAP permutation-explainer importances (``shap_attribution``).
``shap`` is imported lazily so callers without it installed get a clean
error, and the permutation path remains the default everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quant_fund.research.explainability.scoring import (
    PredictFn,
    ProperScoreSpec,
    resolve_score,
)


@dataclass(frozen=True)
class FeatureAttribution:
    """Attribution for one feature under a proper score."""

    feature: str
    importance_mean: float
    importance_std: float
    rank: int  # 1 = most important


@dataclass(frozen=True)
class AttributionResult:
    """Ranking of features by proper-score degradation."""

    method: str
    scoring: str
    baseline_score: float
    attributions: tuple[FeatureAttribution, ...]  # sorted by importance_mean desc
    n_rows: int
    n_repeats: int
    seed: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def top(self, k: int) -> list[FeatureAttribution]:
        return list(self.attributions[: max(0, int(k))])

    def importance_vector(self, feature_names: list[str]) -> NDArray[np.float64]:
        """Mean importances aligned to ``feature_names`` order (missing -> 0.0)."""
        by_name = {a.feature: a.importance_mean for a in self.attributions}
        return np.asarray([by_name.get(name, 0.0) for name in feature_names], dtype=float)


def _validate_xy(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).ravel()
    if xx.ndim != 2 or xx.shape[0] == 0:
        raise ValueError("x must be a non-empty 2-D array")
    if xx.shape[1] == 0:
        raise ValueError("x must contain at least one feature")
    if xx.shape[0] != yy.shape[0]:
        raise ValueError(f"x/y length mismatch: {xx.shape[0]} vs {yy.shape[0]}")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("x and y must be finite")
    return xx, yy


def _rank_attributions(
    feature_names: list[str],
    means: NDArray[np.float64],
    stds: NDArray[np.float64],
) -> tuple[FeatureAttribution, ...]:
    """Sort by mean importance desc; ties break by feature name for determinism."""
    order = sorted(range(len(feature_names)), key=lambda j: (-float(means[j]), feature_names[j]))
    out = []
    for rank, j in enumerate(order, start=1):
        out.append(
            FeatureAttribution(
                feature=feature_names[j],
                importance_mean=float(means[j]),
                importance_std=float(stds[j]),
                rank=rank,
            )
        )
    return tuple(out)


def permutation_attribution(
    predict: PredictFn,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str] | None = None,
    *,
    scoring: ProperScoreSpec | str | None = None,
    n_repeats: int = 5,
    seed: int = 42,
    max_rows: int | None = None,
) -> AttributionResult:
    """Permutation importance of each column of ``x`` under a proper loss.

    For feature ``j`` the column is shuffled ``n_repeats`` times with a
    deterministic ``numpy.random.Generator(seed)`` stream; importance is the
    mean increase in the loss over the baseline. Equivalent protocol to
    sklearn's ``permutation_importance`` (Breiman 2001).

    ``max_rows`` subsamples the eval set without replacement before scoring
    (same subsample across all features and repeats — keeps deltas honest and
    runs bounded on wide panels).
    """
    xx, yy = _validate_xy(x, y)
    n_rows, n_features = xx.shape
    names = _resolve_feature_names(feature_names, n_features)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    spec = resolve_score(scoring)
    rng = np.random.default_rng(int(seed))
    warnings: list[str] = []

    eval_idx = np.arange(n_rows)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        if max_rows < n_rows:
            eval_idx = rng.choice(n_rows, size=max_rows, replace=False)
            warnings.append(f"subsampled {max_rows} of {n_rows} eval rows")
    x_eval = xx[eval_idx]
    y_eval = yy[eval_idx]

    baseline = float(spec(y_eval, predict(x_eval)))

    deltas = np.zeros((n_features, n_repeats), dtype=float)
    for j in range(n_features):
        for rep in range(n_repeats):
            perm = rng.permutation(eval_idx.shape[0])
            x_perm = x_eval.copy()
            x_perm[:, j] = x_eval[perm, j]
            deltas[j, rep] = float(spec(y_eval, predict(x_perm))) - baseline
    if not np.isfinite(deltas).all():
        raise ValueError("permutation score deltas must be finite")

    means = deltas.mean(axis=1)
    stds = deltas.std(axis=1, ddof=1) if n_repeats > 1 else np.zeros(n_features)
    return AttributionResult(
        method="permutation",
        scoring=spec.name,
        baseline_score=baseline,
        attributions=_rank_attributions(names, means, stds),
        n_rows=int(eval_idx.shape[0]),
        n_repeats=int(n_repeats),
        seed=int(seed),
        warnings=tuple(warnings),
    )


def shap_attribution(
    predict: PredictFn,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str] | None = None,
    *,
    scoring: ProperScoreSpec | str | None = None,
    seed: int = 42,
    max_rows: int = 64,
    max_evals: int | None = None,
) -> AttributionResult:
    """Optional SHAP path: mean |SHAP value| per feature via the permutation explainer.

    Uses ``shap.explainers.Permutation`` with an independent masker over the
    eval frame. SHAP values are prediction-space; the reported ``scoring``
    field keeps the proper-score context and ``baseline_score`` stays the
    proper loss of the unmodified eval set (SHAP importances are a
    supplementary view, not a loss decomposition).

    ``y`` participates only for the baseline score. Raises ImportError with
    an actionable message when ``shap`` is unavailable.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap
        raise ImportError(
            "shap backend requested but shap is not installed; "
            "use permutation_attribution or `uv add --no-sync shap`"
        ) from exc

    xx, yy = _validate_xy(x, y)
    n_rows, n_features = xx.shape
    names = _resolve_feature_names(feature_names, n_features)
    spec = resolve_score(scoring)
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    rng = np.random.default_rng(int(seed))

    eval_idx = np.arange(n_rows)
    if max_rows < n_rows:
        eval_idx = np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))
    x_eval = xx[eval_idx]
    y_eval = yy[eval_idx]
    baseline = float(spec(y_eval, predict(x_eval)))

    explainer = shap.explainers.Permutation(
        predict,
        x_eval,
        feature_names=list(names),
        max_evals=int(max_evals) if max_evals is not None else max(2 * n_features + 1, 11),
        seed=int(seed),
    )
    try:
        explanation = explainer(x_eval)
    except AttributeError as exc:
        from quant_fund.compute.numba_guard import compiler_defect

        if not compiler_defect(exc):
            raise
        # Numba 0.68 on CPython 3.13 raises get_call_template inside shap's
        # masker. The permutation attribution is the same proper-score view.
        fallback = permutation_attribution(
            predict,
            x_eval,
            y_eval,
            names,
            scoring=spec,
            n_repeats=1,
            seed=int(seed),
            max_rows=int(x_eval.shape[0]),
        )
        return AttributionResult(
            method="shap_permutation_numba_fallback",
            scoring=fallback.scoring,
            baseline_score=fallback.baseline_score,
            attributions=fallback.attributions,
            n_rows=fallback.n_rows,
            n_repeats=fallback.n_repeats,
            seed=fallback.seed,
            warnings=fallback.warnings
            + (
                "shap permutation explainer hit a Numba compiler defect; "
                "fell back to permutation attribution",
            ),
        )
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(f"unexpected shap value shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("shap values must be finite")
    means = np.abs(values).mean(axis=0)
    stds = np.abs(values).std(axis=0)
    return AttributionResult(
        method="shap_permutation",
        scoring=spec.name,
        baseline_score=baseline,
        attributions=_rank_attributions(names, means, stds),
        n_rows=int(eval_idx.shape[0]),
        n_repeats=1,
        seed=int(seed),
        warnings=(
            "shap importances are mean |value| in prediction space, not a loss decomposition",
        ),
    )


def _resolve_feature_names(feature_names: list[str] | None, n_features: int) -> list[str]:
    if feature_names is None:
        return [f"x{j}" for j in range(n_features)]
    names = [str(name) for name in feature_names]
    if len(names) != n_features:
        raise ValueError(f"feature_names length {len(names)} does not match x columns {n_features}")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names
