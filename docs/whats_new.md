# What's new

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
