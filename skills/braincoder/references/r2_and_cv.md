# R² and cross-validated R²: comparing models without fooling yourself

The lessons here come from comparing nested PRF and modulation-field models on multi-subject
fMRI data (fits as in `convergence.md`). Terms are labelled generically; the lessons do not
depend on the specific effects.

## Save SSE and SST, then choose the summary

Save per voxel and fold: `sse_test`, `sst_test`, plus the train equivalents. Clipped R²
cannot be pooled afterwards.

| Summary | Formula | Favours |
|---|---|---|
| Per-voxel mean | mean over voxels of `ΔSSE_v / SST_v` | low-variance voxels (small SST inflates them) |
| Pooled | `Σ_v ΔSSE_v / Σ_v SST_v` | high-variance voxels; what the optimizer minimised |

**They can disagree in sign.** We have seen the same term, on the same voxels, improve the
per-voxel mean ΔR² in most subjects while worsening the pooled ΔR² in most subjects. Decide on
one metric in advance, report it with the percentage of subjects, and treat a sign flip as a
warning that a few voxels drive the result. Use pooled when comparing models, since it matches
the loss.

### Pooled R² can be set by a handful of voxels: check first

Percent-signal-change scaling divides by each voxel's mean, so voxels with a very low raw
mean (mask edge, signal dropout) can have enormous variance. Pooled R² then describes those
voxels and nothing else. In one subject fitted without any voxel selection, a single voxel
held about half of the total test variance and ten voxels most of it; with them, pooled cvR²
was near zero and the model ordering reversed, and dropping those ten restored the expected
ordering.

- This looks like runaway parameters and is not: those voxels' amplitudes are large because
  their scale is large. Compute the shares before blaming the optimizer.
- Such voxels typically have a low raw mean.

**High variance alone doesn't make a voxel bad.** Tabulate fit quality and cross-run
reliability by voxel SD before choosing a cutoff. In our data, voxels with moderately high SD
had the *best* signal and cross-run reliability, and only a small extreme tail was
irreproducible. A tight SD cutoff therefore removes good data and can weaken a real effect;
a loose cutoff that removes only the unreliable tail does not.

**Rule:** report the top-1 and top-10 share of `Σ SST` for every subject × ROI. Exclude only
voxels that are both extreme and irreproducible, using a criterion that doesn't depend on any
encoding model, and show that the conclusion survives more than one cutoff. Selecting voxels
by a fit-based score doesn't solve it: within a few hundred selected voxels per ROI, one voxel
can still hold a large share of the variance.

**Shared parameters are exposed too.** Per-voxel parameters are independent, so these voxels
only distort the summary. A parameter shared across voxels (a modulation width or gain) is
fitted to summed squared error, so such voxels can dominate its fit. Screen them before
fitting shared-parameter models.

## Restrict the comparison to voxels with signal

Use a **model-free reliability** selection: the null below, held out across runs, > 0. It
does not favour any encoding model. A large share of voxels can fail it, and they give
extreme ΔcvR².

- Stimulus terms help most where the null is worst, because the null lacks the stimulus
  that differs between runs. This is a design property: check what your null can capture.
- Signal voxels can become rare in frontal and parietal ROIs. Show n per ROI and don't
  interpret a median over a few subjects.

## Model-neutral voxel selection for model comparison

This is the rule used in our single-trial projects. Which voxels enter the comparison must not
favour any of the models being compared, and the held-out data must play no part in choosing them.

**Never select on one model's own score.** Selecting voxels by "the richer model beats the
null" favours the richer model, and selecting on the simplest nested model tilts the test the
other way. We have seen the selection alone decide which model wins, even though the score
itself was held out.

**The rule** (per fold, on training data only, scored on the held-out run):

1. Fit every model to the real data and to K label shuffles, with exactly the same procedure,
   including grid, nested starts and gradient descent. Each shuffle is one relabelling seen by all
   models. For single-trial betas, shuffle rows within run. The shuffles can then be stacked as
   extra voxels in one fit, because each voxel is independent.
2. Per model, compare training R² with that model's own pooled null:
   `p_m(v) = (1 + #{null R²_m ≥ R²_m(v)}) / (1 + pool)`. A null copy never enters its own pool.
3. Take `T(v) = max_m −log10 p_m(v)`: a voxel qualifies if *any* model beats its null. Because each
   model is standardised against its own null, a flexible model gets no head start.
4. Select within each ROI with permutation FDR (SAM step-up, `π0 = 1`), then apply a shuffle-level
   gate: count discoveries `D` for each copy treated as data, and keep the ROI's selection only if
   `(1 + #{D_k ≥ D_0}) / (1 + K) ≤ .05`. This needs K + 1 ≥ 20 copies; with few shuffles per
   fold, pool the null copies across folds.
5. For each fold, take each model's median held-out cvR² over that fold's voxels, then average over
   folds. Test paired differences between models across participants.

**Why each part is there:**

- **The gate:** one relabelling lifts R² in many voxels at once, so voxels aren't independent
  units; the shuffle is the unit of exchangeability. Without the gate, min-p + BH made
  discoveries in pure-noise shuffles far more often than the nominal rate. The gated rule's
  bound holds whatever the voxel dependence.
- **Per fold:** selecting on full-data fits inflates the winning model's advantage.
- **Check the selection itself:** treat each shuffle as data, take the top 5% of voxels by `T`, and
  confirm which model supplies the maximum and that the real held-out Δ cvR² there matches that over
  all voxels. Show the neutrality each time.
- **Report sensitivity rules** (min-p + BH at two q values, and the winning model's own null) next
  to the primary rule, plus the number of participants with any selection. A gate can remove a
  whole participant × ROI.

Use a separate rule for description and decoding, such as the model's own null with BH per area,
because those analyses are no longer comparing models.

**Limits:**

- **Cost:** K shuffles × folds × models. Bank at most about 2,000 voxels per ROI to keep it affordable.
- **Timeseries:** the row shuffle and stacking above assume single-trial betas. For BOLD timeseries,
  shuffling rows destroys autocorrelation, so the null would not match the data, and we have no
  validated alternative. When runs share a stimulus sequence, use a **model-free reliability
  selection** instead: within each fold's training runs, predict each run by the mean timecourse
  of the others, and keep voxels with R² > 0. It needs no fits and can't favour an encoding model.
  It has no calibrated false-positive rate, and it misses voxels that respond only to what differs
  between runs, so don't describe it as equivalent to the permutation rule.
- **Shared parameters:** stacking is invalid when voxels share a parameter.
- **It doesn't deal with extreme-variance voxels:** when scoring by pooled R² you still need the check
  above. The median over voxels is much less exposed.

Code: `braincoder.selection` (`PermutationSelector`, `NullBank`) exists only on the `keras-backend`
branch, not in 0.6.2. The gated rule is not part of braincoder (it is a few lines of numpy).

## Which null

| Null | Prediction | Use |
|---|---|---|
| Training mean | one number per voxel; cvR² ≈ 0 | floor |
| **Cross-run mean response** | mean BOLD over training runs per within-run TR, when runs share a stimulus sequence | reproducibility ceiling, voxel selection |

A training mean computed with `.mean(axis=0)` over all TRs instead of per TR gives
cvR² ≈ 0 and a false "no signal in any voxel".

## Choose the fold by the question

| Scheme | Held out | Tests | Watch out |
|---|---|---|---|
| Leave one condition out | all runs of a condition | transfer to an unseen condition | A condition-specific term must generalise to a condition it never saw; penalty concentrates where the term acts |
| Leave one run per condition out | one run of each condition | prediction with all conditions in training | Folds mix conditions, so per-voxel results can't be split by condition unless re-predicted per run |
| Leave one subject out | no fit on the subject | is the *population value* useful | See below |
| Any | | | **Warm starts must not have seen held-out data:** rerun the seed chain inside each fold |

The scheme can change the answer: small terms that lose within-subject cross-validation can
still help when their population value is applied to a left-out subject (below), while large
stimulus-driven terms tend to win under every scheme.

## A real effect can lose cross-validation

Adding a term with true value θ and per-subject estimation variance σ²_θ changes the expected
held-out SSE by about `(σ²_θ − θ²)·ΣX²`. **If |θ| < its per-subject SE, the term hurts out of
sample even when it's real**, and its penalty shows up exactly where its regressor has its
energy.

So don't conclude "not in the data" from a within-subject CV loss. Check the ratio first,
then use a test with no estimation noise for that subject.

## Leave-one-subject-out, zero-parameter test

1. Fit the simpler model to each subject.
2. Compute the term's regressor: `X = ∂pred/∂θ` by central difference. Check linearity.
3. Save per voxel `a = ΣX²`, `b = ΣX·residual`, `sst`. Then `ΔSSE(δ) = δ²a − 2δb`.
4. For subject s, set δ to the median over the *other* subjects of `Σb/Σa`.
5. For modulation fields, set the width to the other subjects' median in the same way. The
   others' optima were computed at widths that include this subject's, a small leak; state it.
6. Test ΔR² across subjects (pooled per subject × ROI, signal voxels).

**Take δ from the conditional optimum `b/a`, not from a parent-and-child joint refit.** A
joint fit lets other parameters compensate, which inflates the term. Since `ΔSSE < 0`
requires `|δ| < 2|b/a|`, an inflated δ overshoots and can turn a small gain into a loss.

The result is a statement about the population, not about prediction for one subject's own
fit. Report it as such, with the effect size, which can be tiny.

## Time-series CV with moving stimuli (bars)

For leave-one-run-out comparisons on bar-mapping data:

- **Pooled cvR² = 1 − SSE/SST** on the held-out run, with test data and predictions
  **demeaned per run**, so the training-mean null is 0 by construction. A time-shifted
  prediction is a weak null (it goes negative).
- **Report cvR² per test sweep direction.** When training and test bars sweep in opposite
  directions, position and HRF delay are confounded, and cvR² can be negative for every method.
- **Concatenating training runs leaks**: the HRF tail of one run falls into the next. Reset
  the HRF per run, or state the leak.
- **Select vertices on training runs only** (for example on grid in-sample R²) and use the
  **same vertices for all models**.
- Free-HRF bounds get hit: count the vertices at a bound.
- **Check the aperture orientation** against an independent map (for example the dataset
  authors' maps): r(y, y0) must be strongly positive; a strongly negative r(y, y0) means a
  flipped aperture.
- The stop rule is on the mean R² improvement over 100 iterations (below 1e-4), so it is
  deterministic: see `performance.md`.

## Quick reference

- `r2 == 1.0` exactly: phantom from mismatched indices, not a good fit. `get_rsq` raises on
  this since 0.6.2, so recompute any cvR² from older code. See `SKILL.md`.
- `r2 == -1000` or all-zero parameters (before 0.6.2), NaN (since 0.6.2): the voxel has a NaN
  sample or zero variance. Drop these before aggregating, and count how many you dropped.
- Clipping at −1 hides how bad a voxel is. Fine for a map, not for aggregation.
- In-sample R² always favours the more flexible model. Never compare models on it, and
  beware "signal voxel count at FDR" between models: it compares thresholds, not prediction.
- Also check n per ROI after every aggregation: a subject that OOM'd can vanish silently.
