# Changelog

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
