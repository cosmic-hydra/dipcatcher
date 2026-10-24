# P6.2 statistical-core audit — `src/quant_fund/metrics/`

Census: 108 modules (`quality/audit_coverage.json` directory `metrics`).
`_bootstrap_kernels.py` is the numba stationary-bootstrap fill, bit-identical
to the NumPy reference and covered by `tests/unit/metrics/test_hotpath_floors.py`.

Scope: `inference.py`, `snooping.py`, `bootstrap.py`, `scoring.py`,
`calibration_tests.py`. Each function was checked against the method cited in
its docstring (formula, degrees of freedom, multiple-testing correction, block
length, seed determinism). Verdicts: `correct`, `fixed` (with failing test),
`suspicious-but-unproven`, `unverifiable`. Reproducing/known-answer tests live in
`tests/unit/metrics/test_stats_audit.py` (45 tests, deterministic, offline,
synthetic data only).

## Bugs fixed (each in its own commit)

1. `optimal_block_length` — **FIXED**. Implementation now faithfully matches
   Politis–White 2004 / Patton–Politis–White 2009: `K_N = max(5, ceil(log10 n))`,
   `m_max = ceil(sqrt n) + K_N`, `cv = 2 sqrt(log10 n / n)`, circular
   autocorrelation `R(k) = sum x_i x_{i+k} / n` on restricted lag-1 vectors,
   flat-top weights `lambda(k/M)` with `M = min(2 max(m_hat,1), m_max)`,
   `b = (2 G^2 / D_SB)^(1/3) n^(1/3)` where `D_SB = 2 (sum_k lam_k R_k)^2`.
   Previous code used `c_i = 1` for the circular scheme and omitted the `m_max`
   cap/`B_max` bound, returning e.g. 21.38 instead of 18.14 on AR(1)
   phi=0.5 n=4000, and collapsing to 1.0 for dominant negative
   autocorrelation (G<0 guard). Verified identical to
   `arch.bootstrap.optimal_block_length(...).stationary` to ~1e-14 for
   values >= 1.
2. `coverage` — **FIXED**. Non-finite observations were counted as interval
   misses (downward-biased coverage) while every sibling scorer masks them;
   now masks non-finite triples and returns NaN when nothing remains
   (fail-closed convention).
3. `bootstrap_sharpe_ci` — **FIXED**. Constant return series produced
   Sharpe ~ 8.9e16: `np.std` of a constant float array returns ~1.8e-18,
   slipping past the `sd <= 0` guard and minting an astronomical statistic —
   the exact failure mode this codebase documents against in
   `newey_west_variance`. Now fails closed to (NaN, NaN, NaN) when
   `std <= 1e-15`, matching the module's degenerate-input convention.
4. `two_way_clustered_mean_tstat` — **FIXED** (second pass). The Cameron–
   Gelbach–Miller 2011 third term must be the variance clustered on the
   (a, b) intersection cells; the code used the White/observation-level
   diagonal `sum resid^2 / n^2`. The two coincide only when every (a,b)
   cell is a singleton; with multi-observation cells the discrepancy can
   raise or lower the t-stat (not a conservative approximation). Now
   clusters on unique (a,b) pairs. KAT: 8-observation panel with cell sums
   -5, 3, -3, 5 -> V = 68/64 = 1.0625 (vs the buggy 94/64), t = 4.5/sqrt(V);
   singleton-cell KAT unchanged (V = 31/36).

## inference.py

| function | method cited | verdict | evidence |
|---|---|---|---|
| `newey_west_variance` | Newey–West 1987 Bartlett HAC | correct | KAT: lags=0 gives gamma0/n = 0.3125 on [1,2,3,4]; lags=1 gives 0.390625; Bartlett weights `1-j/(l+1)` verified against hand computation |
| `newey_west_se` | sqrt of NW | correct | sqrt wrapper; NaN propagation verified |
| `mean_tstat` | NW-HAC t, t(n-1) df | correct | KAT t=4.4721, p=0.020835 on [1,2,3,4]; t(n-1) is conservative vs asymptotic N — documented choice |
| `grouped_mean_tstat` | cluster at group level then NW | correct | KAT on 3 group means [1.5,3.5,5.5]: t=3.7123, df=2 |
| `overlap_aware_hac_lags` | Hansen–Hodrick horizon rule | correct | `max(1.5 n^{1/3}, h-1)`; float-floor artifact at perfect cubes (n=1000 -> 14 not 15) — benign, off-by-one on a heuristic default |
| `two_way_clustered_mean_tstat` | Cameron–Gelbach–Miller 2011 two-way | **fixed** | Third term now clusters on the (a,b) intersection cells (V_{a∩b}); previously used the White/diagonal variance — equal only for singleton cells. KAT on a non-singleton panel pins V = 68/64 exactly; singleton-cell KAT still matches V = 31/36. Remaining note: no `G/(G-1)` finite-sample factor |
| `wild_cluster_bootstrap_two_way_p` | cluster-robust wild bootstrap | suspicious-but-unproven | Rademacher weights applied within level-`a` clusters only; not a true Menzel-style two-way wild bootstrap. Documented as a research approximation |
| `diebold_mariano` | Diebold–Mariano 1995 / HLN small-sample | correct | KAT: d=[1,-1,2,0], lags=0 -> stat 0.8944, p=0.4370; tie handling, `preferred` mapping, lags passthrough verified |
| `onesided_from_twosided` | standard | correct | KAT: (2.0,0.1)->0.05 greater, (-2.0,0.1)->0.95 |
| `mean_difference_t` | paired t with NW | correct | KAT: se=sd/sqrt(n)=0.2 -> t=5.0, p=2.48e-6 |
| `two_proportion_test` | pooled z / Fisher exact | correct | KAT: 40/100 vs 20/100 -> z=3.0861, p=0.002028; sparse-cell Fisher path verified in existing edges test |
| `benjamini_hochberg` | Benjamini–Hochberg 1995 | correct (minor conservatism) | KAT: [0.001,0.02,0.04,0.5,0.9] at 0.05 -> reject ranks 1-2, cutoff 0.02; NaN p-values count toward m in the denominator — conservative, cannot inflate rejections |
| `circular_block_indices` | circular block bootstrap (PP1994) | correct | KAT: block=n yields cyclic shifts, each row a permutation |
| `stationary_bootstrap_indices` | Politis–Romano 1994 stationary bootstrap | correct | geom(p=1/b) block lengths; unbiasedness of resample mean verified; mean_block=1 reduces to iid with uniform marginals |
| `optimal_block_length` | Politis–White 2004 | **fixed** — see above | now matches `arch` reference to 1e-14 |
| `bootstrap_mean_ci` | stationary bootstrap percentile CI | correct | constant -> degenerate CI; iid width ~ 2·1.96·sd/sqrt(n) verified; percentile (not BCa) as documented |
| `bootstrap_sharpe_ci` | stationary bootstrap CI on Sharpe | **fixed** — see above | constant series -> NaN triple, not 8.9e16 |
| `jobson_korkie_memmel` | Jobson–Korkie 1981 / Memmel 2003 | correct | delta-method derived independently: denom (1/n)[2(1-rho)+0.5(sr_b^2)(1+0.5 sr_a^2 - 2 rho)]; KAT theta=4.0825 |
| `pairwise_diebold_mariano` | DM over column pairs | correct | wiring over `diebold_mariano`; diagonal/self pairs handled |

## snooping.py

| function | method cited | verdict | evidence |
|---|---|---|---|
| `reality_check` | White 2000 | correct | V = max_k sqrt(T) fbar_k on recentered stationary-bootstrap means; p = (1+exceed)/(B+1) Davison–Hinkley. KAT: constant panel -> stat sqrt(T), p = 1/(B+1) exactly; strong winner -> minimal p |
| `spa_test` | Hansen 2005 SPA | correct | Verified line-by-line identical to `arch.bootstrap.SPA` (existing reference test `test_snooping_reference.py` compares on identical draws): studentized t = fbar/sigma with bootstrap sigma of sqrt(T) fbar; g_lower = max(fbar,0), g_consistent mask `fbar >= -sigma sqrt(2 log log T / T)`, g_upper = fbar; monotone `p_lower <= p_consistent <= p_upper` verified |
| `stepm` | Romano–Wolf 2005 step-down | correct | sorted descending by studentized stat; adjusted p_j = max over suffix of bootstrap exceedance fractions (step-down monotonicity verified along returned order); rejection is the prefix with adj p <= alpha |
| `model_confidence_set` | Hansen–Lunde–Nason 2011 | correct | range statistic max_i,j |t_ij| on recentered draws; running-p monotone rule stops elimination once p >= alpha — verified identical-column panel -> all p=1.0, dominated column -> p = 1/(B+1) exactly |
| `_resolve_block` | PW2004 per-column | correct-as-heuristic | median of per-column PW2004 blocks; shared block is required for paired bootstrap validity (documented) |
| `_studentized_inputs` | Hansen 2005 | correct | bootstrap-variance sigma rather than Hansen's kernel estimator — asymptotically equivalent; existing arch-reference test confirms |
| `_prepare` | — | correct | row-NaN drop, constant-column drop, fail-closed counts |

## bootstrap.py

| function | method cited | verdict | evidence |
|---|---|---|---|
| `wild_bootstrap_residuals` | Mammen 1993 two-point | correct | a=(1+sqrt5)/2, b=(1-sqrt5)/2, p=(sqrt5-1)/(2 sqrt5); empirical moments E[v]=0, E[v^2]=1, E[v^3]=1 verified at 2000 draws |
| `rademacher_bootstrap` | Liu 1988 / Davidson–Flachaire | correct | +-1 symmetric weights; support verified |
| `pairs_bootstrap` | pairs/iid resampling | correct | on exact linear relation every resampled lstsq recovers beta within 1e-14 |
| `sieve_bootstrap_residuals` | Buhlmann 1997 AR sieve | correct | Yule–Walker AR(p) fit verified (phi recovers 0.7 on planted AR(1) within seed tolerance); innovations resampled iid — standard when residuals are model residuals; init uses first p centered residuals rather than burn-in — documented simplification |
| `subsample_statistic` | Politis–Romano 1994 subsampling | suspicious-but-unproven | KAT on arange(10), block 5 -> sub_stats [2..7], q025=2.125, q975=6.875 exact. Flagged: returns the raw subsample distribution — a valid subsample CI needs tau_b/tau_n rescaling; this is a primitive, not a CI constructor (as documented) |
| `circular_block_indices` | Politis–Romano 1992 circular blocks | correct | contiguous wrapped blocks, uniform start; block=n -> permutations verified |

## scoring.py

| function | method cited | verdict | evidence |
|---|---|---|---|
| `pinball_loss` / `mean_pinball` | Koenker–Bassett 1978 | correct | KAT: tau=0.5 -> MAE/2; tau=0.9 asymmetric values 0.9/0.1 |
| `coverage` | empirical interval coverage | **fixed** — see above | masks non-finite triples; NaN when empty |
| `interval_width` | mean interval width | correct | KAT 2.5; inverted intervals masked |
| `quantile_crossing_rate` / `rearrange_quantiles` | Chernozhukov et al. 2010 | correct | KAT: [1,3,2] row -> 1.0 crossing; rearranged -> 0.0 |
| `crps_from_quantiles` | quantile-CRPS integral | suspicious-but-unproven | left Riemann sum `2 sum_k L_tau_k (tau_k - tau_{k-1})`, tau_0 = 0 — under-covers the [tau_K, 1] tail (weights sum to tau_K < 1 -> downward bias vs true CRPS). Convention is **frozen by docs/MATH_SPEC.md** and pinned by `tests/regression/test_scoring_frozen.py`; flagged, not fixed. KAT verifies the frozen formula exactly (0.75 on the point-mass fixture) |
| `crps_gaussian` / `mean_crps_gaussian` | closed-form Gaussian CRPS (Gneiting) | correct | KAT: CRPS(N(0,1),0) = (sqrt2-1)/sqrt(pi) = 0.233695; sigma-multiplicativity identity verified |
| `crps_student_t` / `mean_crps_student_t` | closed-form Student-t CRPS | correct | nu -> infty converges to Gaussian (rtol 5e-3 at nu=500); nu <= 2 -> NaN (variance undefined) verified |
| `log_score_gaussian` / `mean_` | Gaussian log-score | correct | formula checked; sigma <= 0 -> NaN |
| `crps_gaussian_mixture` | Grimit et al. 2006 mixture CRPS | correct | single-component reduces to `crps_gaussian` to 1e-12; mixture terms `w_i w_j A(mu_i-mu_j, sig_i^2+sig_j^2)` verified against formula |
| `gaussian_mixture_quantiles` | bisection on mixture CDF | correct | single-component -> Phi^{-1}(tau) sigma + mu; +-12 sigma bracket adequate (documented) |
| `crps_empirical` | energy-score/CRPS identity | correct | KAT: [0,2] at y=1 -> 0.5; singleton -> |x-y|. Uses 1/n^2 kernel (biased-consistent form) rather than fair 1/(n(n-1)) — documented |
| `qlike` | Patton 2011 QLIKE | correct | KAT: y=hat y -> 0.0; ratio 2 -> 1 - ln 2; strict loss when off |
| `date_level_equal_weight` | — | correct | deterministic date grouping, equal weights |
| `nonoverlapping_origin_mask` | Hansen–Hodrick overlap control | correct | positions k*h kept; verified on mask fixture |
| `overlap_aware_qlike` | QLIKE on nonoverlap subsample | correct | wiring over the two helpers; empty subsample fails closed |
| `one_step_density_summary` | coverage/pinball/CRPS bundle | correct | wires the audited primitives; masking consistent |
| `name_level_qlike` / `name_level_one_step_density_summary` | per-name aggregation | correct | unique (name,date) requirement enforced; per-name masks verified |
| `pearson_ic` / `rank_ic` | IC / Spearman IC | correct | KAT +-1.0 on monotone; average-rank ties verified; constant -> NaN |
| `icir` | mean/std of IC series | correct | KAT: alternating +-1 -> 0.0; constant nonzero -> NaN (sd=0) |
| `pit_values` | PIT via quantile-grid interpolation | correct | KAT: knots 0.5/0.375/0.0/1.0; below/above grid clamps; duplicate quantiles right-continuous |
| `fissler_ziegel_loss` / `mean_fissler_ziegel` | Fissler–Ziegel 2016 FZ0 joint (VaR,ES) | correct | strictly consistent at the true pair: planted uniform-loss grid verifies the (VaR_0.9, ES_0.9) argmin beats all one-sided perturbations — the elicitability check |

## calibration_tests.py

| function | method cited | verdict | evidence |
|---|---|---|---|
| `spiegelhalter_z` | Spiegelhalter 1986 | correct | z = sum((y-p)(1-2p)) / sqrt(sum((1-2p)^2 p(1-p))) verified by hand; KAT: num=0 fixture -> z=0, p=1.0; p=0.8-hit-5/5 -> z=-1.118, p=0.2636; degenerate variance raises (fail-closed) |

## Cross-cutting checks

- Seeds: every stochastic function routes through `np.random.default_rng(seed)`;
  same seed -> identical results verified (SPA repeat-call equality).
- Degrees of freedom: t(n-1) / t(G-1) / chi2 conventions all verified against
  the cited estimators; `two_way_clustered` df = min(na,nb)-1 flagged above.
- Multiple-testing: BH (correct — step-up rank-cumulation includes all ties
  at the boundary automatically), StepM step-down (correct), MCS elimination
  (correct), SPA recentering (correct vs arch).
- No function mints a headline Sharpe/P&L claim; all degenerate inputs fail
  closed or return NaN per the module convention.

## Second pass (P6.2 re-audit)

A fresh independent audit re-verified every verdict above; one flagged item
was proven and fixed (`two_way_clustered_mean_tstat`, bug 4). Additional
verifications and observations:

- `crps_student_t` verified to ~1e-12 against quadrature of the quantile
  integral definition (nu in {2.5, 5, 10, 30}), beyond the nu->inf Gaussian
  limit checked in the first pass.
- `jobson_korkie_memmel` verified against Monte-Carlo variance of the
  Sharpe-difference on correlated series (ratio ~1.0–1.04, within MC error).
- `spa_test` re-diffed against `arch.bootstrap.SPA`: identical lower/consistent
  recentering thresholds (`-sqrt(2 ln ln T)` on studentized means) and the
  max(0, max t_k) statistic.
- `interval_width` masks inverted intervals while `coverage` counts them as
  misses — deliberate asymmetry: coverage detects malformation, width reports
  well-formed intervals only. Documented, not a bug.
- `subsampling` (`subsample_statistic`), `wild_cluster_bootstrap_two_way_p`,
  and the frozen `crps_from_quantiles` convention remain as documented in the
  table — primitive/approximation/frozen respectively, not fixable without
  breaking the frozen spec or inventing a non-canonical estimator.
- Caller audit: `sweep_research` event-level panels can hit non-singleton
  (event_time, security_id) cells, so fix 4 is material there; all other
  callers (research/agent, benches/families, sota_evidence, net_tournament,
  reality_sweep, analytics) pass proper-score metrics under compliant key
  names — no FORBIDDEN-metric violations found.
