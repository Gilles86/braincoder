# Decoding with braincoder

A posterior is only as good as its noise model. A misspecified Ω can give a reasonable
posterior *mean* and badly overconfident *widths*, so check calibration on held-out
trials before reading posterior SD as "uncertainty". Full argument:
`docs/noise_models.rst` in the braincoder repo.

## Which noise model

| Noise model | Use for |
|---|---|
| `LowRankNoise` (0.6+) | Default when you interpret posteriors and have several training runs. Rank `k` and temperature `T` are chosen by nested, run-grouped CV on the decoding objective. Not shown to beat the baseline below on numerosity or risk data (see the next paragraph) |
| `ResidualFitter` | Reproducing old pipelines and comparing with the literature (van Bergen et al. 2015, Jehee) |
| Diagonal with a tuned `T`: `LowRankNoise.fit(..., k=0)` | **Required baseline** for any covariance claim; also the choice for few voxels |

**Always fit a temperature, and always include the independent-voxel baseline
(`LowRankNoise.fit(..., k=0)`, `T` tuned).** In five datasets (numerosity, risk,
orientation) most of the gain of any covariance model over independent voxels was the
temperature. At `T = 1` the Jehee covariance (`ResidualFitter`) gave the best held-out log
score and independent voxels the worst; with `T` tuned, low rank (k and T by CV) did not beat
independent voxels on log score in any numerosity or risk row (low rank minus independent
was −0.006, −0.042 and −0.200 nats per trial in neuralpriors, tmsrisk and risk7t). Low rank helped only for
orientation (MAE and r), where Jehee with tuned `T` did as well. Do not claim that low rank
improves decoding or behaviour prediction without that contrast. Synthetic checks agree that
CV is a weak selector of structure: with independent truth it chose k = 0 in only 49% of
folds, and with geodesic truth low rank captured only 19 to 41% of the correlation variance.

Why `ResidualFitter` was demoted from default, from the two-numerosity 7T single-trial betas
(retinonumeral, 19 participants, 38 participant × ROI, **joint log score of (n1, n2) relative
to a uniform posterior**, production pipeline at `T = 1`):

- Its Ω has **one** shared factor. Held-out squared Mahalanobis distances were about 4×
  too large (about 16× along decoding directions), all 38 participant × ROI posteriors scored
  worse than a uniform posterior (mean −0.44 nats, range −2.4 to −0.004), and 95% HDI
  coverage was 0.79. That is a result on a two-numerosity task with that scoring. It does
  **not** transfer to the single-numerosity and risk datasets, where in the paired three-way
  comparison (absolute log score, `T = 1`) Jehee was better than low rank and independent
  voxels, so do not quote "worse than flat" for `ResidualFitter` in general.
- **`lambd` shrinkage is unsafe when voxels > trials**: coverage about 0.04 at `lambd=0.5`.
- `get_pseudoWWT` builds WWᵀ from raw predictions, so the common-mode response dominates it.

On simulated data `ResidualFitter` can look fine. In a clean simulation with independent
noise it reached coverage of 0.5 / 0.95. Good simulation results do not transfer here;
check on the real data. Cost also differs: `ResidualFitter` took about 21 s at 200 voxels,
3 min at 1,000, 12 min at 2,000 and is extrapolated at about 23 h (10 GB RAM) at 10,000,
plus about 7 inner fits if you tune `T`; low rank takes 0.5 to 4 s over the same range.

The nested CV chose k=0 in about 60% of folds and T≈360 (median) on that dataset. **T ≫ 1
means there is little calibrated stimulus information**; report it rather than hiding it.
Always report `noise.summary()`.

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
  grid, a stimulus-independent tilt of the prior earns up to 1.4 to 1.8 nats, and the level of
  the score depends on the dataset: compare arms with paired contrasts only, never across
  datasets. Do not compare MAE across `T` settings either (it rose from about 24 to 31 in
  risk rows when `T` was tuned, for a reason not yet explained).

- **`groups=` matters.** Trials within a run share noise; without it, `n_folds`
  contiguous blocks are used.
- **Pass `inner_parameters`** (a dict `{group: parameters}` from existing leave-one-run-out
  fits, or a refit callable). By default the encoding parameters stay fixed in the inner
  folds, and those trials helped fit them. This is not a small bias: the chosen `T` came out
  several-fold too small (median 1.7 to 5.7 against 4 to 90 with refit) and `k` much too
  large (k = 0 chosen in under 3% of folds against 41 to 74% with refit), and the held-out
  log score was 0.1 to 6 nats per trial lower in every participant. The fixed-inner default
  is for exploration only.
- **`k` selected by decoding, not by covariance fit.** The covariance likelihood keeps
  asking for more factors (to k=160) while decoding worsens beyond k≈40.
- Training stimuli are matched to the nearest grid point, with a warning if they are
  more than half a step away. Use a fine enough grid.
- **Legacy code path:** `omega, dof = fit_lowrank_omega(residuals, k=40)`, then
  `get_stimulus_pdf(data, grid, omega=omega, dof=dof, temperature=T)`, or
  `noise.to_dense()`.

## Decoding BOLD time series (moving bars)

`get_stimulus_pdf` decodes one response at a time: right for betas, wrong for BOLD, because
it ignores the HRF (and a `*WithHRF` model decodes without it). One TR mixes the last
~30 s of stimulus. Findings from CHN (3T) and Szinte (7T) bar data:

- Decode a whole run jointly with the HRF in the forward model, or use a static per-volume
  decoder with the summed HRF and a **lag chosen on training runs** (a lag chosen on the test
  run flatters).
- `LowRankNoise.fit` assumes independent trials, so its CV for k and T does not apply. A
  frame-conditional `T` objective chose T = 11.3 against a known truth (grossly
  underconfident). `T` also absorbs residual autocorrelation (CHN 16 to 64, Szinte 4 to 8).
  Choose k and T on held-out training runs with the decoder you will use.
- The per-frame posterior needs a two-step blank rule (blank only if the blank beats all
  bars summed, else the best bar), because a blank candidate has a much larger prior than any
  single bar.
- Tempering cannot calibrate a sequence posterior (95% coverage peaked at 0.80 over T = 8 to
  128 on the reduced arms); a closed-form per-volume posterior reached 0.94. Sequence MAP
  sat 1,300 to 2,700 nats above the true sequence's objective on real data (misspecified,
  not badly optimised).
- Single-site Gibbs over discrete bar labels gets stuck in whole-sweep width/orientation
  relabelling (R-hat inf) and needs segment, relabel and FFBS moves; even then 95% intervals
  held the truth about half the time. Plain NUTS on soft-edged bars did not converge (the
  tree saturated at 255 steps). Do not use a sequence sampler or NUTS without an R-hat and
  coverage check.
- Validate on a simulation with known pRFs first, and report per-volume posterior coverage
  on held-out runs. A non-negative pixel MAP (ADMM, Kronecker structure) is a TFP-free
  alternative to `StimulusFitter` (frame r 0.994 against it on a small simulation).

Sources: `notes/analyses/decoding_methods.md`, `bar_map_methods.md`, `bar_phrf_methods.md`,
`bar_sampler_methods.md` in the paper_braincoder repo.

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
  so α near 1 means almost all shared noise decays with distance. On numerosity and risk data
  the term did not help at `T = 1` (worse than plain Jehee in every participant, by 0.57 and
  0.92 nats in two datasets); a tuned-`T` version was never run. Do not claim it helps.
- `init_pseudoWWT(subtract_baseline=...)` changes the decoded values (Jehee r 0.094 with the
  default `True`, 0.117 with `False`, the production setting on the same data): fix it and
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
new prior, grid or readout then takes minutes. Without the cache, one wide-prior control
required refitting 32 sessions.

## Checklist before believing a decoding result

0. **Confirm the trial order of the betas before any decoding.** Correlate the beta pattern
   of trial i with the BOLD pattern about 5 s after the onset of trial i + L for L = −3..3;
   the peak must sit at L = 0 in nearly every run (624 of 624 runs in the numerosity data;
   unsmoothed `risk3t` betas failed it, the smoothed ones passed). Also compare GLMsingle's
   `DESIGNINFO.npy` `stimorder` with the behavioural log, and do not join `events.tsv` and a
   behavioural log on `trial_nr` without checking its meaning (1 to 240 per session in one,
   1 to 30 per run in the other). GLMsingle hyperparameters (noise PCs, ridge fraction) saw
   all runs, including held-out ones: a small known leak, state it.
1. The voxel selection and noise model saw training trials only.
2. `parameters=` and `dof=` were passed explicitly, and each fold used its own.
3. The model is non-HRF and fitted on the same kind of data you decode.
4. `calibration_summary` on held-out trials: coverage near nominal, `pit_sd` near 0.289.
5. Shuffled labels give r≈0. r well above 0 under shuffling means leakage.
6. Check whether the result depends on the grid range.
7. For stimuli confounded with something the region encodes (orientation fixing value,
   numerosity fixing area), an above-chance r does not yet identify the dimension.
