# Decoding with braincoder

A posterior is only as good as its noise model. A misspecified Ω can give a reasonable
posterior *mean* and badly overconfident *widths*, so check calibration on held-out
trials before reading posterior SD as "uncertainty". Full argument:
`docs/noise_models.rst` in the braincoder repo.

## Which noise model

| Noise model | Use for |
|---|---|
| `LowRankNoise` (0.6+) | Default when you interpret posteriors and have several training runs. Rank `k` and temperature `T` are chosen by nested, run-grouped CV on the decoding objective. Not guaranteed to beat the baseline below: test it (see the next paragraph) |
| `ResidualFitter` | Reproducing old pipelines and comparing with the literature (van Bergen et al. 2015, Jehee) |
| Diagonal with a tuned `T`: `LowRankNoise.fit(..., k=0)` | **Required baseline** for any covariance claim; also the choice for few voxels |

**Always fit a temperature, and always include the independent-voxel baseline
(`LowRankNoise.fit(..., k=0)`, `T` tuned).** In our experience much of the apparent gain of a
covariance model over independent voxels comes from the temperature: at `T = 1` the ranking
of noise models can look very different from the ranking once `T` is tuned on held-out data,
and with `T` tuned a low-rank covariance does not reliably beat independent voxels. Do not
claim that low rank improves decoding or behaviour prediction without that contrast.
Cross-validation is also a weak selector of covariance structure: in simulations it often
picks the wrong rank, so do not read the chosen `k` as evidence about the true noise
structure.

Why `ResidualFitter` is no longer the default:

- Its Ω has **one** shared factor. On some real datasets this produced strongly
  overconfident posteriors (held-out squared Mahalanobis distances far larger than expected,
  95% intervals covering well under 95% of trials, held-out log scores below those of a flat
  posterior); on others it did well. Whether it is adequate depends on the data, so check
  calibration rather than assuming either way.
- **`lambd` shrinkage is unsafe when voxels > trials**: coverage can collapse.
- `get_pseudoWWT` builds WWᵀ from raw predictions, so the common-mode response dominates it.

On simulated data `ResidualFitter` can look fine: in a clean simulation with independent
noise it reaches nominal coverage. Good simulation results do not transfer here; check on
the real data. Cost also differs: `ResidualFitter` took about 21 s at 200 voxels, 3 min at
1,000, 12 min at 2,000 and is extrapolated at about 23 h (10 GB RAM) at 10,000, plus about
7 inner fits if you tune `T`; low rank takes 0.5 to 4 s over the same range.

**T ≫ 1 means there is little calibrated stimulus information**; report it rather than
hiding it. Always report `noise.summary()`.

## Recipe

```python
from braincoder.noise import LowRankNoise
from braincoder.utils.calibration import calibration_summary, plot_calibration

grid = np.linspace(0, 40, 81).astype(np.float32)     # the prior: see below
noise = LowRankNoise.fit(model, train_data, train_paradigm, pars,   # training data ONLY
                         stimulus_range=grid, groups=train_runs)    # groups = runs
pdf = model.get_stimulus_pdf(test_data, grid, parameters=pars, noise_model=noise)
print(noise.summary())
calibration_summary(pdf, test_stimuli)   # log_score, coverage_0.5/0.95, pit_sd (≈.289), r_posterior_mean
plot_calibration(pdf, test_stimuli)
```

- **Say whether `log_score` is absolute or relative to a flat posterior** (subtract
  `log(1/grid range)`). If the experiment uses only a few stimulus values on a much wider
  grid, a stimulus-independent tilt of the prior can earn a substantial score on its own,
  and the level of the score depends on the dataset: compare arms with paired contrasts
  only, never across datasets. Do not compare MAE across `T` settings either; it can move
  in either direction when `T` is tuned.

- **`groups=` matters.** Trials within a run share noise; without it, `n_folds`
  contiguous blocks are used.
- **Pass `inner_parameters`** (a dict `{group: parameters}` from existing leave-one-run-out
  fits, or a refit callable). By default the encoding parameters stay fixed in the inner
  folds, and those trials helped fit them. This is not a small bias: the chosen `T` comes out
  too small and `k` too large, and the held-out log score suffers. The fixed-inner default is
  for exploration only.
- **Select `k` by decoding, not by covariance fit.** The covariance likelihood keeps
  asking for more factors long after decoding has started to worsen.
- Training stimuli are matched to the nearest grid point, with a warning if they are
  more than half a step away. Use a fine enough grid.
- **Legacy code path:** `omega, dof = fit_lowrank_omega(residuals, k=40)`, then
  `get_stimulus_pdf(data, grid, omega=omega, dof=dof, temperature=T)`, or
  `noise.to_dense()`.

## Decoding BOLD time series (moving bars)

`get_stimulus_pdf` decodes one response at a time: right for betas, wrong for BOLD, because
it ignores the HRF (and a `*WithHRF` model decodes without it). One TR mixes the last
~30 s of stimulus. Advice for bar-mapping data:

- Decode a whole run jointly with the HRF in the forward model, or use a static per-volume
  decoder with the summed HRF and a **lag chosen on training runs** (a lag chosen on the test
  run flatters).
- `LowRankNoise.fit` assumes independent trials, so its CV for k and T does not apply. `T`
  also absorbs residual autocorrelation, so expect it to be well above 1. Choose k and T on
  held-out training runs with the decoder you will use, and check the objective against a
  simulation with known truth (a frame-conditional objective can pick a grossly
  underconfident `T`).
- The per-frame posterior needs a two-step blank rule (blank only if the blank beats all
  bars summed, else the best bar), because a blank candidate has a much larger prior than any
  single bar.
- Tempering may not be able to calibrate a sequence posterior; a closed-form per-volume
  posterior is easier to calibrate. A sequence MAP that scores far better than the true
  sequence on real data signals misspecification, not a poor optimiser.
- Single-site Gibbs over discrete bar labels can get stuck in whole-sweep width/orientation
  relabelling and needs segment, relabel and FFBS moves. Plain NUTS on soft-edged bars may
  not converge (watch for tree-depth saturation). Do not use a sequence sampler or NUTS
  without an R-hat and coverage check.
- Validate on a simulation with known pRFs first, and report per-volume posterior coverage
  on held-out runs. A non-negative pixel MAP (ADMM, Kronecker structure) is a TFP-free
  alternative to `StimulusFitter`.

## With `ResidualFitter`

```python
model = GaussianPRF()
fitter = ResidualFitter(model, train_data, train_paradigm, parameters=pars)
omega, dof = fitter.fit(stimulus_range=grid, method='t', init_dof=10., max_n_iterations=1000)
pdf = model.get_stimulus_pdf(test_data, grid, parameters=pars, omega=omega, dof=dof)
```

- With weights, pass `weights=` to the fitter instead of `stimulus_range=`. Image models need neither.
- `fitter.fitted_omega_parameters` holds `tau, rho, sigma2, dof`; `fitter.n_failed_steps`
  counts rolled-back steps. Look at them. Ω = ρ·ττᵀ + (1−ρ)·diag(τ²) + σ²·WWᵀ is
  regularized, so raw-sample-covariance rank worries (T ≲ V) do not apply to it. ρ near 1
  is the near-singular case.
- It raises `ValueError` on NaN residuals (drop those voxels) or an empty voxel selection, and
  `RuntimeError` after 50 consecutive failed steps.
- `use_wwt=False` strips the σ²·WWᵀ term (and hurts; keep it); `spherical=True` fits a
  single-scale Ω.
- `D=` adds a distance-dependent (geodesic) term
  (`examples/01_decoding_pipeline/02_geodesic_decoder.py`). In **0.7.0 and earlier** the
  kernel is not guaranteed positive semi-definite on a graph distance and beta is
  unconstrained; 0.7.1 computes `exp(-beta D)` on a classical-MDS embedding
  (`mds_embed_distances`, O(n²) memory), constrains beta > 0 and alpha in (0, 1), and raises
  for `D` combined with `lambd > 0` or `spherical=True`. Distances come from a
  nearest-vertex mapping (no check that a voxel lies in cortex; two voxels on one vertex
  have D = 0), so check that voxels lie in cortex. Ω = ρ[α(e^(−βD̃)∘ττᵀ) + (1−α)ττᵀ] + …,
  so α near 1 means almost all shared noise decays with distance. In our tests the term did
  not help; do not claim it helps without a held-out comparison against plain
  `ResidualFitter` and the tempered independent-voxel baseline.
- `init_pseudoWWT(subtract_baseline=...)` changes the decoded values noticeably: fix it and
  report it.
- `get_stimulus_pdf(omega=…)` uses `safe_cholesky` with adaptive jitter, so a slightly non-PSD Ω is fine.

## The grid is the prior

`stimulus_range` is the support of a flat prior, with no separate prior argument.

- **Truncation fakes precision and bias.** In a test, narrowing the grid from 0–40 to
  15–25 halved the posterior SD (0.85 → 0.43). Stimuli near the edge get truncated
  posteriors, and expected-value decoding regresses toward the center. Extend the grid
  beyond the stimulus range when you interpret width or bias, or test explicitly.
- A wide-prior control only needs `get_stimulus_pdf` to be rerun if Ω and the parameters
  are cached (below).
- Normalization integrates over the grid value, so a non-uniform grid remains a flat
  density on the stimulus axis.
- 2D stimuli return an unnormalized table unless you request
  `include_multidimensional_stimulus_index=True`; use `utils.calibration.marginalize` to
  get marginals.

## Cache the intermediates

Write Ω (or the `LowRankNoise` parts), `dof`, weights and the per-fold parameters to
disk, keyed by `(subject, session, fold, model)`, in addition to posterior summaries. A
new prior, grid or readout then takes minutes. Without the cache, a single wide-prior
control can mean refitting every session.

## Checklist before believing a decoding result

0. **Confirm the trial order of the betas before any decoding.** Correlate the beta pattern
   of trial i with the BOLD pattern about 5 s after the onset of trial i + L for L = −3..3;
   the peak must sit at L = 0 in nearly every run. If it does not, find out why before
   decoding. Also
   compare GLMsingle's `DESIGNINFO.npy` `stimorder` with the behavioural log, and do not join
   `events.tsv` and a behavioural log on `trial_nr` without checking its meaning (it can
   count per session in one file and per run in the other). GLMsingle hyperparameters (noise
   PCs, ridge fraction) saw all runs, including held-out ones: a small known leak, state it.
1. The voxel selection and noise model saw training trials only.
2. `parameters=` and `dof=` were passed explicitly, and each fold used its own.
3. The model is non-HRF and fitted on the same kind of data you decode.
4. `calibration_summary` on held-out trials: coverage near nominal, `pit_sd` near 0.289.
5. Shuffled labels give r≈0. r well above 0 under shuffling means leakage.
6. Check whether the result depends on the grid range.
7. For stimuli confounded with something the region encodes (orientation fixing value,
   numerosity fixing area), an above-chance r does not yet identify the dimension.
