---
name: braincoder
description: >-
  braincoder: Python/Keras 3 library for fitting fMRI encoding models (population receptive fields for visual space,
  numerosity, value, orientation) and inverting them for Bayesian stimulus decoding. Step-by-step manual, silent traps
  (Keras backend, index alignment, grid and scale, *WithHRF, version differences), noise models and calibration, voxel
  selection, cvR², runtimes. Read before writing or debugging code touching `braincoder`, `ParameterFitter`,
  `ResidualFitter`, `LowRankNoise`, `GaussianPRF*`, `get_stimulus_pdf` or `KERAS_BACKEND`, or when a PRF fit is slow,
  out of memory, or gives implausible R².
---

# braincoder

## What it is

braincoder is a Python library that answers two questions about fMRI data:

1. **Encoding:** how does each voxel respond to a stimulus? You describe the stimulus,
   choose a tuning model (for example a Gaussian over visual space or over number), and
   braincoder estimates each voxel's tuning.
2. **Decoding:** given a new brain response, which stimulus most likely caused it?
   braincoder inverts the fitted model with Bayes' rule. That gives a full probability
   distribution over stimuli on every trial, so you get a best guess *and* an uncertainty.

The second part is what sets it apart from most PRF packages, which stop at the tuning
maps. The trial-by-trial uncertainty is what lets you link neural precision to behaviour.

It runs on TensorFlow, JAX or PyTorch through Keras 3, so one fit can use a GPU and
handle a whole cortical surface: 93,534 vertices took 6–13 minutes on a recent GPU in the
paper benchmark (median 351 s on an H200 with JAX to 752 s on an A100 with TF).

| Use it for | Look elsewhere for |
|---|---|
| Tuning to a continuous stimulus: position, number, value, orientation | Plain contrasts and mass-univariate GLMs (nilearn, SPM) |
| Trial-wise decoding with calibrated uncertainty | Single-trial betas themselves (GLMsingle, then bring them here) |
| Your own tuning function, without writing an optimizer | Non-fMRI data: untested for anything else so far |
| Whole-brain PRF mapping on a GPU | Cognitive models of behaviour (**bauer**) |

- **Cite:** de Hollander, Renkert, Ruff & Knapen, *braincoder*, Zenodo,
  [10.5281/zenodo.10778413](https://doi.org/10.5281/zenodo.10778413). Papers, users,
  people: [references/about.md](references/about.md).
- **Install / code:** <https://github.com/Gilles86/braincoder> (step 1 below).
- **Docs:** <https://braincoder-devs.github.io/> (tutorial lessons 1–7; can lag a release).

This file describes **0.7** (`__version__ = '0.7.0'` on `main`): models hold configuration
only, and old constructor calls raise an error
([migration guide](https://braincoder-devs.github.io/migration.html)). **0.7.1 is being
released** and fixes four silent problems that exist in 0.7.0 and earlier: the `SPMHRFModel`
kernel (every `*WithHRF` fit), a memory leak in torch gradient descent, `ResidualFitter(D=)`
geodesic noise, and NaN in narrow HRFs. Until you have 0.7.1, read those rows in
[references/version_history.md](references/version_history.md) before trusting an HRF fit.
Project pins may be older: for results from an older checkout, see the same file.
`python scripts/check_skill_claims.py` re-tests every claim here against your installed
version.

## Manual: a new project in seven steps

[`scripts/quickstart.py`](./scripts/quickstart.py) is all of this as one script on
simulated data. It runs in about a minute: run it first, then swap in your data.

**1. Install.** Pin a commit (or a release tag once one exists for 0.7; the latest tag on
GitHub is `v0.6.2`), so that your results stay reproducible:

```bash
pip install "braincoder @ git+https://github.com/Gilles86/braincoder.git@<commit>" tensorflow
```

Use `tensorflow[and-cuda]` on a GPU node, and one env per backend. Add this to every job:

```bash
export KERAS_BACKEND=tensorflow
python scripts/preflight.py --backend tensorflow --require-gpu   # prints what will actually run
```

**2. Prepare two tables** with the same index:

- `data`: one row per trial or TR, one column per voxel.
- `paradigm`: one row per trial or TR, the stimulus value in each column.
- Single-trial betas (GLMsingle): confirm that row i really is trial i before any decoding
  (a shuffled or shifted order still gives above-chance decoding in noisy data). Check:
  [`references/decoding.md`](./references/decoding.md), checklist item 0.

```python
data = data.loc[:, data.notna().all() & (data.std() > 0)]   # no NaN or empty voxels
assert data.index.equals(paradigm.index)                     # rows must line up
```

**3. Choose a model.**

| Your data | Your stimulus | Model |
|---|---|---|
| Single-trial betas | A number on a linear axis | `GaussianPRF` |
| Single-trial betas | Numerosity or value | `LogGaussianPRF` |
| Single-trial betas | Orientation or direction | `AxialVonMisesPRF` or `VonMisesPRF`; with few trials (about 100) a fixed basis + `WeightFitter`, see [Models](#models) |
| BOLD timeseries | Moving bars, visual field | `GaussianPRF2DWithHRF` |
| BOLD timeseries | Visual field with surround | `DifferenceOfGaussiansPRF2DWithHRF` or `DivisiveNormalizationGaussianPRF2DWithHRF` |

More in [Models](#models) below.

**4. Fit, in three stages.** A coarse grid search finds the rough tuning, a regression
sets amplitude and baseline, and gradient descent polishes:

```python
import numpy as np
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter

fitter = ParameterFitter(GaussianPRF(), data, paradigm)
grid = fitter.fit_grid(mu=np.linspace(5, 35, 16), sd=np.linspace(3, 15, 7),
                       amplitude=[1.], baseline=[0.], use_correlation_cost=True)
init = fitter.refine_baseline_and_amplitude(grid)
pars = fitter.fit(init_pars=init, max_n_iterations=3000, progressbar=False)
```

Grid ranges should span plausible values, with the `sd` floor no narrower than you can
believe. Keep the stimulus on a scale of roughly 1–50.

**5. Check the fit on data it has not seen.** In-sample R² (`fitter.r2`) always flatters:

```python
from braincoder.utils import get_rsq
cv_r2 = get_rsq(test_data, GaussianPRF().predict(test_paradigm, pars))
```

Fit on all runs but one, test on the left-out run. Compare with a null model that
predicts each voxel's training mean. Also simulate data from known parameters and check
that you recover them at your real noise level.

**6. Select voxels on the training data only:**

```python
from braincoder.utils.stats import fit_r2_mixture, r2_fdr_threshold
keep = train_r2 > r2_fdr_threshold(fit_r2_mixture(train_r2), .05)
```

**7. Decode the left-out run, and check the uncertainty is honest:**

```python
from braincoder.noise import LowRankNoise
from braincoder.utils.calibration import calibration_summary

stimulus_grid = np.linspace(0, 40, 81).astype(np.float32)    # extend past the stimulus range
model = GaussianPRF()                                        # never a *WithHRF model here
noise = LowRankNoise.fit(model, train_data[keep], train_paradigm, pars[keep],
                         stimulus_range=stimulus_grid, groups=train_runs)
posterior = model.get_stimulus_pdf(test_data[keep], stimulus_grid,
                                   parameters=pars[keep], noise_model=noise)
calibration_summary(posterior, test_stimuli)   # 95% interval should cover about 95% of trials
```

**Where to run it:** an ROI fit runs on a laptop or CPU. Above a few thousand voxels, use a
GPU. Loops over a whole cohort belong on the cluster, even if each subject takes under a
minute. Save parameters, weights and Ω per fold, so that a new readout needs no refit
(below). Write brain maps as float32 (rule 14).

**Saving results.** 0.7 has no save/load helper and models hold no fitted state, so save
what you pass to the methods, per fold:

```python
pars.to_csv(f'{out}/pars_fold-{run}.tsv', sep='\t')                  # index = voxel labels
np.savez(f'{out}/noise_fold-{run}.npz', scale=noise.scale, factors=noise.factors,
         psi=noise.psi, dof=np.nan if noise.dof is None else noise.dof,
         temperature=noise.temperature, voxels=np.asarray(noise.voxels))
# reload
z = np.load(f'{out}/noise_fold-{run}.npz', allow_pickle=True)
noise = LowRankNoise(z['scale'], z['factors'], z['psi'], dof=float(z['dof']),
                     temperature=float(z['temperature']), voxels=z['voxels'])
```

The noise round-trip reproduces the posterior exactly (checked on 0.7.0). Also record the
braincoder version/commit, backend, model class and its constructor
arguments (e.g. `grid_coordinates`, `tr`) next to the files.

## Where to read next

| About to | Read |
|---|---|
| Learn the concepts from scratch | [Tutorial lessons](https://braincoder-devs.github.io/tutorial/index.html) |
| Decode, or choose a noise model | [`references/decoding.md`](./references/decoding.md) |
| Size a job, pick a GPU or backend, set iterations, debug slowness or OOM | [`references/performance.md`](./references/performance.md) |
| Write a new model class | [`references/custom_models.md`](./references/custom_models.md) |
| Fit BOLD timeseries in the visual field (2D `*WithHRF`) | [Timeseries PRFs](#timeseries-prfs-2d-withhrf) below |
| Decode a BOLD time series (moving bars) rather than single-trial betas | [`references/decoding.md`](./references/decoding.md), "Decoding BOLD time series" |
| Compare braincoder with prfpy, popeye, analyzePRF, prfmodel, GEM-pRF | [`references/performance.md`](./references/performance.md), "Comparing packages" |
| Reuse results from an older braincoder | [`references/version_history.md`](./references/version_history.md) |
| Change braincoder itself | [`references/library_development.md`](./references/library_development.md) |
| Understand a model's equation and parameters, or `rf_normalization` | [`references/models_and_parameters.md`](./references/models_and_parameters.md) |
| Tell a converged fit from optimizer noise; runaway parameters; nested models | [`references/convergence.md`](./references/convergence.md) |
| Compare models with R² or cvR²; choose folds, voxels and a null; a CV loss vs a group effect | [`references/r2_and_cv.md`](./references/r2_and_cv.md) |
| Measure speed on a new env | [`scripts/bench_fit.py`](./scripts/bench_fit.py) and `bench_fit.sh` |

## The rules that prevent silent errors

Almost every failure here is silent: the fit finishes and the numbers look plausible.
Read these before you trust a result.

**1. Know which braincoder you are running.** Each project pins its own `libs/braincoder`,
and a cluster env may import a different checkout from the one you are reading. Versions
in use range from pre-Keras TF forks to the current release, and several older ones have
silent bugs ([references/version_history.md](references/version_history.md)).
`scripts/preflight.py` prints the version and path and can fail on a mismatch.

**2. Set `KERAS_BACKEND` in every job, before the first keras import.** Otherwise
`~/.keras/keras.json` decides. With `"jax"` there and a CPU-only jaxlib, a job on a GPU
node ran at 25–70 s/iteration while TF reported the GPU, so a "does TF see the GPU" check
passed.

**3. `data.index` must equal `paradigm.index`.** `get_rsq` raises on different labels, but
the fitters pair rows **by position** (warning if a labelled index differed), so a wrong
row order is still your job to prevent. Assert it (step 2).

**4. Remove NaN voxels before fitting.** A voxel with a single NaN sample, or zero variance,
cannot be fitted. It returns NaN parameters and NaN `fitter.r2`, with a warning, and
decoding with such a row raises `ValueError`. Masked voxels still cost compute (21% of the
vertices in one surface dataset).

**5. Always grid, then refine, then fit.** The correlation-cost grid leaves amplitude and
baseline at your placeholder, so its raw R² is about 0 (−0.004 → 0.43 after refine). Use a
nonzero amplitude in the grid: `[0.]` makes all predictions constant, and refine raises
`not invertible` on TF or returns junk on JAX.

**6. The grid decides what width you get.** Gradient descent rarely travels far from the
grid, so a grid that offers needle widths returns needles. Raising the grid's `sd` floor
fixed one such collapse, with no need for `fixed_pars` or a prior.

**7. Keep the stimulus on a scale near 1–50, or raise the learning rate.** Adam moves about
`learning_rate` per step, so 1000 steps at 0.01 move `mu` by at most 10. Multiplying the
stimulus by 100 cut the movement from 4.8 to 0.1 in original units. Rescale a 0–900 value axis.

**8. Decode betas with the plain model, not a `*WithHRF` model.** A `*WithHRF` model decodes
without the HRF and warns, so its amplitude must be on the scale of your betas: parameters
fitted to BOLD timeseries may not be. `get_fisher_information` and
`get_expected_uncertainty` refuse `*WithHRF` models.

**9. Pass `omega` and `dof` together.** `dof=None` means a Gaussian likelihood, whatever you
fitted.

**10. Models hold configuration only.** Pass `paradigm`, `parameters`, `weights`, `omega` and
`dof` to each method; leaving one out raises an error. One model object can safely serve many
folds. Fitters do keep their results (`fitter.r2`, `fitter.estimated_parameters`).

**11. Parameter order differs between models, so call `fit_grid` with keywords.** 1D is
`mu, sd, amplitude, baseline`; 2D is `x, y, sd, baseline, amplitude`.

**12. `WeightFitter` regresses through the origin by default.** A Gaussian basis can absorb a
DC offset; an all-positive linear feature gets a biased slope and can decode at r≈0. Use
`weights, intercept = fitter.fit(fit_intercept=True)`, or demean the data.

**13. To link the neural code to behaviour, simulate; don't use Fisher information.** Simulate
from the fitted model and noise model, then decode with an ideal Bayesian observer.
`get_expected_uncertainty` does this for 1D stimuli.

**14. Write NIfTIs through a mask with `set_data_dtype(np.float32)` and
`header.set_slope_inter(1, 0)`**, or parameters quantize to 256 values through the `uint8`
mask. See **cogneuro-project**.

## Models

| Stimulus | Class | `parameter_labels` |
|---|---|---|
| 1D, linear axis | `GaussianPRF` | `mu, sd, amplitude, baseline` |
| 1D, compressive axis | `LogGaussianPRF` | same, or `mode, fwhm, …` with `parameterisation='mode_fwhm_natural'` |
| Circular; π-periodic orientation | `VonMisesPRF`; `AxialVonMisesPRF` | `mu, kappa, amplitude, baseline` |
| Skewed | `AlphaGaussianPRF` | `mu, sd, alpha, amplitude, baseline` |
| Visual field | `GaussianPRF2D` | `x, y, sd, baseline, amplitude` |
| Polar | `GaussianPRF2DAngle` | `theta, ecc, sd, baseline, amplitude` |
| Surround | `DifferenceOfGaussiansPRF2D` | `x, y, sd, baseline, amplitude, srf_amplitude, srf_size` |
| Normalization | `DivisiveNormalizationGaussianPRF2D` | `x, y, sd, rf_amplitude, srf_amplitude, srf_size, neural_baseline, surround_baseline, bold_baseline` |
| Parameters depend on condition | `RegressionGaussianPRF`, `RegressionAlphaGaussianPRF` | one column per (parameter, regressor) |
| Fixed basis, fit weights only | `LinearModel`, or a PRF with fixed `parameters` + `WeightFitter` | |

- **Orientation with few trials (about 100 per participant): fit weights on a fixed basis**
  (for example 8 half-wave-rectified cos⁵ channels) with `WeightFitter(..., fit_intercept=True)`,
  not per-voxel `AxialVonMisesPRF`. On the same 200 voxels the basis model decoded 3 to 5
  degrees better (MAE 24.3 vs 28.4 and 26.9 vs 30.1 in two experiments) and was better
  calibrated (the per-voxel model's 95% coverage was 0.62 to 0.87). Score circularly
  (circular mean of the posterior, circular MAE) with `axial` periodicity (180 degrees).
- **Single-trial betas take a non-HRF model; timeseries take a `*WithHRF` model** with
  `hrf_model=SPMHRFModel(tr=…)`. Mixing them up gives no error.
  `flexible_hrf_parameters=True` appends `hrf_delay, hrf_dispersion`. Put
  `assert np.argmax(SPMHRFModel(tr=1.).get_hrf()) == 5` in every job: in 0.7.0 and earlier
  the kernel peaked at 6 s (see [Timeseries PRFs](#timeseries-prfs-2d-withhrf)).
- **2D models:** the paradigm is `(n_t, n_pixels)` and `grid_coordinates` is
  `(n_pixels, 2)`. An aperture not flipped to match the coordinate convention still fits
  well, but lands mirrored, so check one visual-field map.
- **`rf_normalization`** rescales amplitudes only: `'legacy'` (default), `'peak'`
  (prfpy's `normalize_RFs=False`), `'area'`. It matters **when you simulate**: `amplitude=1`
  gives predicted SD 0.003 in legacy and 0.04 in peak on a 20 × 20 grid, so `noise=.05`
  swamps legacy (x recovered at r=0.40) but not peak (r=0.99).
- **`allow_neg_amplitudes=True`** on 1D models, with `positive_amplitude=False` in the grid
  and refine, allows troughs. Analyse them apart from peaks, or they cancel when averaged.
- **`noise_model`** stays `'ssq'` by default. `'gaussian'` has the same per-voxel optimum, helps from a
  poor start and not from a good grid, and costs about 25% per iteration; see
  [`references/performance.md`](./references/performance.md).

### Timeseries PRFs (2D `*WithHRF`)

The 2D path differs from the seven-step manual in three places: the paradigm is the
stimulus aperture movie, the model needs `grid_coordinates` and an HRF, and the parameter
order is `x, y, sd, baseline, amplitude`. From `examples/00_encodingdecoding/fit_prf.py`
(0.7):

```python
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter

# paradigm: aperture per TR, (n_volumes, n_pixels) or (n_volumes, H, W) (flattened for you)
# grid_coordinates: (n_pixels, 2) DataFrame/array of x, y in degrees, same pixel order
# data: (n_volumes, n_voxels) BOLD, same index as paradigm
model = GaussianPRF2DWithHRF(hrf_model=SPMHRFModel(tr=tr),
                             grid_coordinates=grid_coordinates.astype(np.float32))
fitter = ParameterFitter(model=model, data=data, paradigm=paradigm)
grid = fitter.fit_grid(x=np.linspace(-8, 8, 10), y=np.linspace(-4, 4, 10),
                       sd=np.linspace(0.1, 4, 10), baseline=[0.], amplitude=[1.],
                       use_correlation_cost=True)
init = fitter.refine_baseline_and_amplitude(grid)
pars = fitter.fit(init_pars=init, max_n_iterations=1000)
```

`braincoder.utils.data.load_szinte2024()` gives a ready-made example dataset (`stimulus`,
`grid_coordinates`, `v1_timeseries`, `tr`). Add `flexible_hrf_parameters=True` to fit the HRF
too (start from the fixed-HRF fit).

**The canonical SPM HRF (peak 5 s) can be about 2 s too slow for real data** (3T CHN and 7T
Szinte data both prefer a delay of about 4 s). Before a free per-voxel HRF, try **one
participant-level HRF** chosen on the training runs (scan delay × dispersion, grid R²): it
gave 0.048 of the 0.059 cvR² gain of the free HRF (82%). Delay is well identified, shape is
not: `peak_s` = delay − dispersion is a gamma mode, not a measured peak, so report a fitted
HRF as an effective latency, and check the optimum is not at the edge of the scan (dispersion
hit it in 20 of 24 fits). Time-series CV with moving bars has its own traps:
[`references/r2_and_cv.md`](./references/r2_and_cv.md). Decoding a BOLD time series is not
`get_stimulus_pdf` on one TR: [`references/decoding.md`](./references/decoding.md).

**Basis-function workflow:** rows of `parameters` and `weights` are basis functions, and
columns of `weights` are voxels:

```python
centers = pd.DataFrame({'mu': np.linspace(0, 40, 9), 'sd': 6., 'amplitude': 1., 'baseline': 0.}).astype(np.float32)
model = GaussianPRF()
weights = WeightFitter(model, centers, data, paradigm).fit(alpha=1.)
predictions = model.predict(paradigm, centers, weights)
```

## Voxel selection

**Select inside the training fold.** Selecting on R² from data that includes test trials
makes decoding circular. **To compare models, select model-neutrally:** any model beats its own
permutation null, never one model's own score. Recipe in
[`references/r2_and_cv.md`](./references/r2_and_cv.md).

- `fit_r2_mixture` fits two Gaussians on logit(R²), dropping NaN and non-positive values; it
  needs at least 50 usable voxels. `r2_fdr_threshold` returns `inf` if nothing reaches α.
  `r2_posterior_signal` gives per-voxel P(signal) and is gentler on small ROIs.
- Use this mixture. Earlier methods were anti-conservative (F-test FDR) or went U-shaped and
  gave near-zero thresholds (Beta mixture).
- **cvR² favours few-parameter models.** A negative median may be just the df cost of the
  model. Compare with a label-permutation null before reporting single-voxel tuning.
- Drop NaN voxels before fitting and report how many (93,534 of 118,584 remained in the
  benchmark).
- A small ROI can select 0 voxels in a fold, and `ResidualFitter` then crashes. Handle it
  explicitly, for example by keeping the top voxel.

## Other pieces

- **Stimulus reconstruction** (`from braincoder.optimize import StimulusFitter`): always pass
  `l2_norm` (start at 0.01, raise it if maps are noisy), or decoded values diverge. It needs
  TensorFlow and TFP, and the import is guarded, so without them you get `ImportError`. Adding
  TFP to a clean Keras 3 env can pull in a second TF (`undefined symbol:
  Wrapped_PyInit__pywrap_*`).
- **GP spatial prior** (`braincoder.optimize.bayesian_fitter.BayesianParameterFitter`,
  `braincoder.optimize.gp_prior.GeodesicGPPrior`; not exported at package level). Empirical
  Bayes after Daghlian et al. 2025. The Cholesky scales as O(V³), so use it on ROIs. What
  worked on numerosity data: a prior on the tuning peak `mu` only, with the
  hyperparameters fitted jointly with the parameters (type-II MAP, which adds log det K to
  the loss), was the best variant, but at full n (39) its gain in decoding r on unsmoothed
  betas is small and not reliable (+0.005 to +0.014, every 95% CI includes 0) and MAE gets
  WORSE by 0.3–0.5 (interim n = 23 had suggested +0.025). Priors on `sd`,
  amplitude or baseline hurt (amplitude and baseline are voxel-specific scale and offset
  with no smoothness rationale; a joint prior on all four was catastrophic on smoothed
  data), and fitting the lengthscale by MLE on noisy per-voxel estimates over-smooths (about
  25 mm, the size of the ROI). Keep the σ²·WᵀW term in the decoder Ω. **Caution:** the
  prior is zero-mean on the native parameter, so by construction it also pulls `mu` toward 0.
  That is design reasoning, not a measured effect, but it fits the observed pattern (MAE
  worse by 0.3 to 0.5 numerosity units with any `mu` prior while r rose slightly); a
  constant-mean variant is under test. Results and status:
  `notes/analyses/gp_prior_summary.md` in the paper_braincoder repo.
- **Not released:** `get_acuity`, `braincoder.datasets.load_dataset` and the stimulus
  parameterisations are on feature branches (`feature/acuity`, `feature/datasets`,
  `feature/stimulus-parameterization`); this skill does not describe them.
- **Examples:** `examples/01_decoding_pipeline/` is the current end-to-end pipeline.
  `docs/noise_models.rst` and `docs/fisher_information.rst` explain the choices.

## Working on the library itself

Coding rules for braincoder contributors (keras.ops, testing on three backends, no
`float()` in a loss): [references/library_development.md](references/library_development.md).
