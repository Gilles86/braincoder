# Changelog

## 0.6.0 (DRAFT — unreleased)

The headline of 0.6 is a new **per-voxel Gaussian likelihood** option in
`ParameterFitter`, opt-in here and the default in 0.7. Same fixed points
as the existing sum-of-squares loss, much faster convergence under
heteroskedastic noise — which is to say, on every real fMRI dataset.

The rest of 0.6 is a substantial set of stability fixes, a new GP-prior
fitter, and a per-fold voxel-selection helper.

### Default fitter changes (heads-up for downstream projects)

- **`ParameterFitter(noise_model={'ssq', 'gaussian'})`** — new knob.
  - `'ssq'` (default in 0.6): unweighted sum-of-squared-residuals.
    Equivalent to MLE under homoscedastic Gaussian noise. Same as 0.5.x.
  - `'gaussian'` (new): per-voxel Gaussian negative log-likelihood with
    a free per-voxel σ²ᵥ. The θ-gradient is `(1/σ²ᵥ)·∂SSQᵥ/∂θᵥ`, so
    each voxel makes comparable progress under the shared Adam optimizer
    regardless of its noise level. Converges to the same per-voxel θ
    at high iterations — but does so much faster in practice on real
    data, where noise varies wildly across voxels.
- **Synthetic verification**: at `max_n_iterations=20_000` both modes
  agree on per-voxel θ to <1e-3 normalized RMSE. At `2_000` iters
  Gaussian mode is at its asymptote; SSQ mode is still ~5× further
  out. See `gp_prior_fmri/notes/analyses/classical_vs_ml_convergence/`
  for the full convergence study.
- **0.7 will flip the default to `'gaussian'`**. If you need exact
  reproduction of pre-0.6 fits, pin `noise_model='ssq'` explicitly.

### Hierarchical Bayesian / GP prior

A new hierarchical-Bayesian fitter for any encoding model, with a
Gaussian-Process prior over cortical geodesic distance.

- **`GeodesicGPPrior`** (`braincoder.optimize.gp_prior`): RBF kernel
  over a precomputed pairwise distance matrix. Adaptive jitter for
  Cholesky stability; classical MDS embedding so the kernel is PSD
  even on non-Euclidean graph distances (Schoenberg 1938); Cholesky
  and triangular solves in float64.
- **`BayesianParameterFitter`** (`braincoder.optimize.bayesian_fitter`):
  three-stage empirical-Bayes (classical → hyperparams via MLE → MAP),
  with options for **type-II joint MAP** (`joint_hyperparams=True`,
  co-fits hyperparameters in stage 3), **shared lengthscale across
  priors** (`shared_lengthscale=True`), and **frozen Cholesky in MAP**
  to keep TF's `CholeskyGrad` out of the backward graph.

### Voxel selection / FDR utilities

- **`fit_r2_mixture` / `r2_fdr_threshold`** (`braincoder.utils.stats`) —
  2-Gaussian mixture on `logit(R²)` with a tail-FDR threshold for
  empirical-null voxel selection.
- **`r2_posterior_signal` / `r2_p_signal_threshold`** — per-voxel
  posterior `P(signal | R²)` and the R² at which it crosses a chosen
  probability. More forgiving than tail-FDR on small/degenerate ROIs.
- **`plot_r2_mixture`** — diagnostic plot of the mixture fit + threshold.

### Fisher information & expected uncertainty

- **Analytical Fisher information for multivariate Student-t noise.**
  `EncodingModel.get_fisher_information(..., analytical=True)` now
  accepts `dof` and applies the closed-form prefactor
  `(ν + p) / (ν + p + 2) · Jᵀ Ω⁻¹ J`. Previously this combination
  raised `ValueError` and forced users to the Monte-Carlo path; MC
  was both noisier (~12% RMS at n=200) and prone to int32 overflow in
  `UnsortedSegmentSum` at `n × n_vox × n_stim ≈ 5e7`. The analytical
  path is now exact for any ν and recovers the Gaussian formula as
  ν → ∞.
- **`EncodingModel.get_expected_uncertainty(stimuli, omega, dof, ...)`**
  — new convenience wrapper. Simulates `n_simulations` noisy responses
  per stimulus from `(parameters, omega, dof)`, decodes each via
  `get_stimulus_pdf`, and returns a DataFrame indexed by stimulus
  value with `mean_E, var_E, mean_error, mean_abs_error, n_sims`.
  `batch_stimuli` lets you bound memory at large grid sizes.
- **Tests**: `tests/test_fisher_information.py` (9 tests): Student-t
  prefactor, dof → ∞ recovery of Gaussian, MC ↔ analytical agreement,
  and the FI ↔ 1/var_E relationship.
- **Docs**: new `docs/fisher_information.rst` page with the two-API
  story, the "is spikiness real?" diagnostic, and a worked example.

### Cortical surface helpers

- **`braincoder.utils.cortex.geodesic_distance_matrix`** — Dijkstra-based
  pairwise geodesic distance on a triangulated mesh, restricted to a
  set of source vertices (typical use: project EPI-mask centroids to
  nearest vertex then run Dijkstra on the matched set).

### Robustness fixes

- **`safe_cholesky(M, jitter=1e-4)`** (`braincoder.utils.backend`) —
  symmetrise + adaptive-jitter wrapper for `ops.cholesky` with a
  retry-on-NaN at 10× jitter. Wired into `get_stimulus_pdf`, Fisher
  information, both `_simulate` noise paths, and the residual fitter
  likelihood. Eliminates the `Cholesky : Tensor had NaN values` class
  of crashes that hit ~10–20% of subjects on production fits.
- **`ResidualFitter.fit(use_wwt=True)`** — when `False`, strips the
  `σ²·WᵀW` tuning-similarity term from Ω. Diagnostic; default
  unchanged.
- **`BayesianParameterFitter.fit_map`** uses a manual NaN-safe global-
  norm gradient clip instead of `keras.optimizers.Adam(clipnorm=...)`,
  which under PyTorch would poison Adam's moment buffers with NaN if
  any single gradient went non-finite.

### Backend layer fixes (TF / JAX / torch parity)

- JAX `compute_gradients` runs inside a `StatelessScope` so
  `keras.Variable` reads inside `value_and_grad` don't leak tracers.
- `sample_mvt` / `sample_student_t` derive independent sub-seeds for
  the χ² and normal components.
- `_lgamma` routes through differentiable backend-native
  lgamma (`tf.math.lgamma` / `jax.scipy.special.gammaln` /
  `torch.lgamma`) instead of the non-differentiable scipy fallback.
- torch `compute_gradients` uses `torch.autograd.grad` instead of
  reading `.grad` off each variable.
- `get_stimulus_pdf` converts user data arrays to tensors before
  `_likelihood` (avoids numpy-on-MPS device-mismatch errors).
- `safe_cholesky` explicitly checks for NaN in the output, since some
  backends return NaN silently rather than raising.

### Tests & CI

- New test files: `test_gp_prior.py` (+360 lines), `test_safe_cholesky.py`
  (+174), `test_backend_multibackend.py` (+376), additions to
  `test_utils.py` (+168).
- `scikit-learn` now declared as a runtime dep (used by
  `fit_r2_mixture`).
- TFP-dependent import test is skipped when `tensorflow_probability` is
  not installed.

### Compatibility

- `numpy.trapz` (deprecated in NumPy 2.0) replaced with
  `numpy.trapezoid` via a shim.

---

## 0.5.1 (2026-05-04)

- Fix invalid `isRelatedTo` relation in `.zenodo.json` (Zenodo schema requires `references` instead). The 0.5.0 GitHub release failed to deposit on Zenodo because of this; 0.5.1 contains no code changes, only the metadata fix.

## 0.5.0 (2026-05-04)

The headline change is a full **Keras 3 port**: braincoder is now backend-agnostic and runs on TensorFlow, JAX, or PyTorch (including Apple Silicon MPS). The codebase has been refactored into subpackages, the test suite has grown from a handful of integration tests to 170 passing tests across all three backends, and a long list of bugs has been fixed.

### Backend & infrastructure

- **Keras 3 multi-backend port**: all core code now uses `keras.ops.*` instead of TensorFlow primitives. TF, JAX, and PyTorch backends are all CI-tested.
- **Subpackage refactor**: the monolithic `models.py` and `optimize.py` are split into `braincoder/models/` (`base`, `linear`, `prf_1d`, `prf_2d`) and `braincoder/optimize/` (`parameter_fitter`, `weight_fitter`, `residual_fitter`, `stimulus_fitter`).
- **GitHub Actions CI** runs the full suite on every push.
- **Test suite** expanded to 170 tests covering models, HRFs, optimizers, stimuli, and utilities.
- **NumPy 2.0 / pandas 3.x compatibility**: `np.trapz` → `np.trapezoid`, `Series[:, np.newaxis]` workaround, `groupby(axis=1)` removed.

### New models

- **`AxialVonMisesPRF`** — π-periodic von Mises PRF for orientation data.
- **Gaussian Mixture PRFs** (`GaussianMixturePRF2D`).
- **Difference-of-Gaussians PRFs** (`DifferenceOfGaussiansPRF2D`, `DifferenceOfGaussiansPRF2DWithHRF`).
- **Divisive Normalization PRFs** (`DivisiveNormalizationGaussianPRF2D`, `DivisiveNormalizationGaussianPRF2DWithHRF`).

### New features

- `subtract_baseline=True` option in `GaussianPRF.init_pseudoWWT` removes per-voxel baseline contamination from the WWT covariance.
- `lambd` (convex blending between parametric and empirical covariance) is now actually wired into `ResidualFitter.fit()` (previously stored but ignored).
- `shared_pars` / `fixed_pars` keep parameters constant across voxels — useful for hard-to-estimate parameters and future attention-field models.
- Multi-dimensional stimulus support, `correlated_response` option, multiple repeats in `simulate()`, configurable HRF parameter ranges.

### Bug fixes

- **`ResidualFitter.lambd`** was silently ignored (closure called the wrong helper).
- **`ResidualFitter` λ=1 edge case**: zero-gradient parametric vars; switched to dof-only short-circuit + adaptive jitter (`1e-4 × mean_diag + 1e-9`) for Cholesky stability.
- **`get_stimulus_pdf`**: pass Cholesky of omega, not omega itself, to `_likelihood`.
- **`get_fisher_information`**: add diagonal jitter before Cholesky.
- **`VonMisesPRF._get_stimulus`** and **`GaussianPRFWithHRF` MRO** init bugs.
- **HRF**: convolution uses `ops.flip` (MPS-compatible); upsample to `highres_dt` for creation; SPM HRF defaults fixed.
- Various amplitude/baseline rescaling, `aggressive_softplus` for `RegressionGaussianPRF`, DN HRF formula correction.

### Cleanup

- Removed dead modules: `braincoder/estimators.py`, `braincoder/tests/`, stub `CustomHRFModel`.
- Removed 5 legacy test files that referenced removed classes.
- Switched from `pkg_resources` to `importlib`.

### Migration notes

- Most user-facing imports are unchanged; subpackage refactor is internal.
- If you were using the old-style monolithic `braincoder.models` module path (e.g. `from braincoder.models import GaussianPRF`), this still works — the names are re-exported.
- For PyTorch on Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` (required by `ops.lstsq` / `ops.solve`).
