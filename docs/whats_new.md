# What's new

## 0.7.0 highlights (2026-11-02)

- **Models no longer hold data or fitted values.** Pass `paradigm`, `parameters`, `weights`,
  `omega` and `dof` to each method. See the [migration guide](migration.rst).
- **New `ParameterFitter.fit` default:** `r2_atol=1e-4`. `noise_model` stays `'ssq'`, contrary to what 0.6
  announced: see the changelog below.
- **On PyPI:** `pip install "braincoder[tensorflow]"`.
- **Torch on Apple GPUs** now runs the whole library.

## 0.6.2 highlights (2026-11-02)

Bug-fix release. Inputs that gave plausible-looking wrong results now give the right result
or a clear error. **Recompute** cvR² from frames with different indices (R² could be exactly
1), **refit** if `data` and `paradigm` had different indices, and **re-decode** anything
decoded with a `*WithHRF` model. Voxels that cannot be fitted now return NaN rather than
zeros and `r2 == -1000`. Details are in the changelog below.

## 0.6.0 highlights (2026-10-25)

- **Calibrated decoding.** {class}`braincoder.noise.LowRankNoise` models the
  noise covariance as low-rank + diagonal (Student-t or Gaussian) and tempers
  the likelihood; rank *k* and temperature *T* are chosen by nested,
  run-grouped cross-validation on the decoding objective. Use it with
  `get_stimulus_pdf(..., noise_model=noise)` (or `temperature=` with a dense
  `omega=`). Guide: [Noise models for decoding](noise_models.rst); worked
  example: [example 04](auto_examples/01_decoding_pipeline/04_lowrank_noise_calibration.rst).
- **Calibration tools.** {mod}`braincoder.utils.calibration`: HDI coverage,
  randomised PIT, HPD level of the truth, log score vs. a flat posterior,
  `calibration_summary` and `plot_calibration`, for any `get_stimulus_pdf`
  output ([API](api_reference.rst)).
- **New fitting options.** `ParameterFitter(noise_model='gaussian')`
  (per-voxel Gaussian likelihood; default stays `'ssq'`), `r2_history_` /
  `loss_history_`, a GP spatial prior (`GeodesicGPPrior`,
  `BayesianParameterFitter`), R² mixture / FDR voxel-selection helpers
  ([example 01](auto_examples/01_decoding_pipeline/01_voxel_selection_fdr.rst)),
  analytical Student-t Fisher information and `get_expected_uncertainty`
  ([Fisher information](fisher_information.rst)).
- **Key fixes.** `safe_cholesky` (adaptive jitter, NaN check);
  `ResidualFitter` no longer swallows errors; differentiable `_lgamma`
  (Student-t ν was biased low on TF and never fitted on JAX — refit Ω/ν
  from older checkouts); TF `compute_jacobian` int32 overflow; MC Fisher
  information on JAX / torch; NumPy 2 `trapezoid`.
- **Documented caveat.** `ResidualFitter`'s one-factor Ω underestimates
  held-out noise on single-trial fMRI; prefer `LowRankNoise` for decoding
  (see [Noise models for decoding](noise_models.rst)).

## Full changelog

```{include} ../CHANGELOG.md
:start-line: 1
:heading-offset: 1
```
