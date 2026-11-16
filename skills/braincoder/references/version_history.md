# braincoder: if your results predate version X, redo Y

SKILL.md states the 0.7 behaviour; **0.7.1 is being released** (not yet on PyPI or tagged; check
`braincoder.__version__` and the commit you actually import) and the first four rows below say
"fixed in 0.7.1". This table is for results produced with an
older checkout. Find out which one you ran with `python scripts/preflight.py` (prints
version and path) or `braincoder.__version__`; projects often pin their own
`libs/braincoder`. Release notes: `docs/whats_new.md` and the
[migration guide](https://braincoder-devs.github.io/migration.html).

| Results made with | Affected | What went wrong (silently) | Redo |
|---|---|---|---|
| ≤ 0.7.0 (fixed in 0.7.1, `fix/spm-hrf` 4b5405c) | any fit with `SPMHRFModel` or a `*WithHRF` model | The HRF peaked at 6 s instead of 5 s (gamma shape was delay/dispersion + 1) and was sampled every 32/(n − 1) s instead of every TR, so timing and amplitude of the convolved prediction were wrong, the fitted `sd` was biased, and bounds on a fitted delay were shifted by 1 | refit. Benchmark on `largegrid`: R² 0.280 → 0.338. Test: `np.argmax(SPMHRFModel(tr=1.).get_hrf()) == 5` |
| ≤ 0.7.0 (fixed in 0.7.1, `fix/torch-gd-leak`) | torch backend, `ParameterFitter.fit` gradient descent | R² tracking ran with autograd on and `best_r2` / `best_parameters` kept the graph: 5 voxel × pixel tensors stay alive per iteration (0, 58, 115, 519, 2,249 MB after iterations 1, 2, 3, 10, 40 at 3,000 voxels × 961 pixels), so large fits ran out of memory (12 benchmark cells at 49,000 and 93,534 voxels on A100/H100/H200). The loss was unchanged, so fits that finished are valid | rerun the OOM jobs; until 0.7.1, use TF or JAX for whole-surface fits |
| ≤ 0.7.0 (fixed in 0.7.1, `feature/noise-geodesic-fixes` 853c5b6, 9e9f14c) | `ResidualFitter(D=...)` geodesic noise | the kernel could be non-PSD on a graph distance and beta had no constraint | refit; see `decoding.md` for what 0.7.1 does and for what the term did on real data |
| ≤ 0.7.0 (fixed in 0.7.1) | `*WithHRF` models with narrow HRFs (gamma shape above about 35) | `gamma_pdf` overflowed in float32 and gave NaN; it now runs in log space | refit the affected voxels |
| JAX backend, checkouts before 2026-10-19 | `ResidualFitter` Ω | returned an **unfitted** Ω | refit Ω, re-decode |
| TF backend, checkouts before 2026-05-21 | `ResidualFitter` ν (`dof`) | ν biased low | refit Ω and ν |
| < 0.6.1 | DN models with `flexible_hrf_parameters=True` | HRF delay followed the BOLD baseline | refit |
| < 0.6.2 | anything where `data.index` ≠ `paradigm.index` | no error: with no overlap `get_rsq` returned R² = 1.0 for every voxel; with partial overlap (`1…n` vs `0…n-1`) `refine_baseline_and_amplitude` compared shifted timeseries (−0.44 where the true R² was 0.43). `fitter.r2` (array-based) stayed correct and hid it | recompute cvR² and refined amplitudes |
| < 0.6.2 | voxels with any NaN or zero variance | got all-zero parameters and `fitter.r2 == -1000` (so `fitter.r2.mean()` meant nothing); a zero row could make every posterior NaN | drop those voxels, recompute summaries and posteriors |
| < 0.6.2 | decoding with a `*WithHRF` model | each candidate stimulus went through the HRF as a one-sample timeseries: tuning curve shrunk ~17× (TR 1.6 s), posteriors ~5× too wide | re-decode with the plain model |
| < 0.6.2 | decoding with Student-t noise | `model.dof` was ignored (Gaussian likelihood) | re-decode passing `dof` |
| < 0.7 | decoding with omitted `omega` | a model could silently fall back on a stored `omega` | re-decode passing `omega` and `dof` explicitly |
| < 0.7 | one model object reused across CV folds | `ResidualFitter` wrote `model.parameters`/`model.weights` and `StimulusFitter` wrote `model.omega`/`model.dof`, so a later fold could decode with the **last** fold's parameters if you omitted them | re-decode with per-fold parameters passed explicitly |
| < 0.7 | comparing new fits to old ones | default convergence tolerance changed | pass `r2_atol=1e-6` to reproduce |
