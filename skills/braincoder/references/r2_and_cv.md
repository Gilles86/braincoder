# R² and cross-validated R²: comparing models without fooling yourself

Examples: a 7T dataset, 30 subjects, fits as in `convergence.md`. Terms are labelled
generically; the lessons do not depend on the specific effects.

## Save SSE and SST, then choose the summary

Save per voxel and fold: `sse_test`, `sst_test`, plus the train equivalents. Clipped R²
cannot be pooled afterwards.

| Summary | Formula | Favours |
|---|---|---|
| Per-voxel mean | mean over voxels of `ΔSSE_v / SST_v` | low-variance voxels (small SST inflates them) |
| Pooled | `Σ_v ΔSSE_v / Σ_v SST_v` | high-variance voxels; what the optimizer minimised |

**They can disagree in sign.** Same term with the same strength, same voxels, ΔR² per
subject × ROI, median over subjects:

| Region | Per-voxel mean (×10⁻³) | Subjects > 0 | Pooled (×10⁻⁵) | Subjects > 0 |
|---|---:|---:|---:|---:|
| V1 | −0.02 | 47% | −0.36 | 7% |
| V2 | **+0.29** | 80% | **−12.6** | 27% |
| V3 | +0.05 | 50% | −1.95 | 37% |
| hV4 | +0.04 | 83% | +0.83 | 60% |
| All (per subject over ROIs) | +0.04 | 83% | −0.15 | 40% |

The per-voxel column is a mean over subjects, the pooled one a median. Decide on one
metric in advance, report it with the percentage of subjects, and treat a sign flip as a
warning that a few voxels drive the result. Use pooled when comparing models, since it
matches the loss.

### Pooled R² can be set by a handful of voxels: check first

Percent-signal-change scaling divides by each voxel's mean, so voxels with a very low raw
mean (mask edge, signal dropout) can have enormous variance. Pooled R² then describes those
voxels and nothing else. One subject, 9,894 voxels in visual and parietal ROIs, all fitted
without any voxel selection:

| Voxels kept (by test SD of the BOLD, % signal) | n | Gaussian | DoG | Gaussian + flex HRF | DoG + flex HRF |
|---|---:|---:|---:|---:|---:|
| All | 9,894 | +0.0070 | **−0.0024** | +0.0068 | **−0.0027** |
| SD ≤ 20 (drops the 10 largest) | 9,884 | +0.1022 | +0.1040 | +0.1067 | +0.1092 |
| SD ≤ 6.6 (99th percentile) | 9,795 | +0.1106 | +0.1136 | +0.1151 | +0.1189 |
| Median voxel, all | 9,894 | +0.0101 | +0.0137 | +0.0113 | +0.0165 |

- The top voxel held 51% of the total test variance and the top 10 held 90% (median voxel SD
  1.1%, largest 418%). Removing the 10 voxels turned a negative R² into 0.10 and restored the
  expected ordering.
- This looks like runaway parameters and is not: those voxels' amplitudes are large because
  their scale is large. Compute the shares before blaming the optimizer.
- The raw mean falls with SD: median 947 for SD ≤ 3, 170 for SD > 10 (mask median 777).

**High variance alone doesn't make a voxel bad.** Tabulate fit quality and cross-run
reliability by SD before choosing a cutoff. In about 111,000 selected voxels from 30 subjects:

| Voxel SD (% signal) | Voxels | Share of variance | Median R² | Median cross-run R² | Reliable |
|---|---:|---:|---:|---:|---:|
| ≤ 1 | 17,801 | 1.6% | .055 | −.050 | 27% |
| 1–2 | 65,703 | 16.6% | .092 | −.011 | 47% |
| 2–3 | 17,243 | 12.1% | .187 | +.109 | 65% |
| 3–5 | 7,659 | 13.2% | .246 | +.172 | 67% |
| 5–10 | 2,412 | 12.9% | .281 | +.193 | 67% |
| 10–20 | 357 | 7.8% | .269 | +.177 | 65% |
| 20–50 | 40 | 3.8% | .124 | −.093 | 43% |
| > 50 | 13 | 31.9% | .069 | −5.2 | 0% |

Voxels up to about 20% have the *best* signal, so a tight cutoff removes good data. A cutoff
of 5% or 3% weakened one term in the example; a cutoff of 20%, which removes only the
unreliable tail, did not. Here 53 of about 111,000 voxels held 36% of the variance.

**Rule:** report the top-1 and top-10 share of `Σ SST` for every subject × ROI. Exclude only
voxels that are both extreme and irreproducible, using a criterion that doesn't depend on any
encoding model, and show that the conclusion survives more than one cutoff. Selecting voxels
by a fit-based score doesn't solve it: in a set of at most 500 voxels per ROI, one voxel still
held a median of 5.7% and up to 67% of the variance.

**Shared parameters are exposed too.** Per-voxel parameters are independent, so these voxels
only distort the summary. A parameter shared across voxels (a modulation width or gain) is
fitted to summed squared error, so such voxels can dominate its fit. Screen them before
fitting shared-parameter models.

## Restrict the comparison to voxels with signal

Use a **model-free reliability** selection: the null below, held out across runs, > 0. It
does not favour any encoding model. Half of all voxels can fail it, and they give extreme
ΔcvR²:

| Null cvR² of the voxel | ≤ 0 | 0–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.3 | > 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| Share of voxels | 52% | 10% | 7% | 10% | 7% | 14% |
| Per-voxel mean ΔcvR² (×10⁻³), stimulus-drive term | +15.4 | +11.3 | +8.3 | +5.0 | +2.7 | +1.4 |
| Per-voxel mean ΔcvR² (×10⁻³), small location-specific term | −3.0 | −0.24 | −0.27 | −0.43 | −0.60 | −0.92 |

- Stimulus terms help most where the null is worst, because the null lacks the stimulus
  that differs between runs. This is a design property: check what your null can capture.
- Signal voxels vanish in frontal and parietal ROIs (V1 83%, V3 75%, IPS 12%, SPL1 1.7%,
  FEF 0.4%). Show n per ROI and don't interpret a median over 3 subjects.

## Model-neutral voxel selection for model comparison

This is the rule used in our single-trial projects. Which voxels enter the comparison must not
favour any of the models being compared, and the held-out data must play no part in choosing them.

**Never select on one model's own score.** In one project, voxels were selected by "richer model
beats the null in cvR²" and that richer model then won in 24 of 24 subjects. Selecting instead
on the simplest nested model, which tilts the test the other way, gave 2 of 24. That is a
complete reversal from the selection alone, and it happened even though the score was held out.

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
  units; the shuffle is the unit of exchangeability. In a per-fold ROI setting, min-p + BH at
  q = .05 made at least one discovery in 22% of pure-noise shuffles. The gated rule made one in
  under 1% there, and its bound holds whatever the voxel dependence.
- **Per fold:** selecting on full-data fits inflated the winning model's advantage over most
  competitors by 10–30% in one comparison.
- **Check the selection itself:** treat each shuffle as data, take the top 5% of voxels by `T`, and
  confirm which model supplies the maximum and that the real held-out Δ cvR² there matches that over
  all voxels. The neutrality held in the example, but it should be shown each time.
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
branch, not in 0.6.2. The gated rule is separate project code for now (pure numpy).

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

Percentage of subjects whose prediction improves for the same six terms:

| Term | Leave one condition out | Leave one run per condition out | Leave one subject out, zero parameters |
|---|---:|---:|---:|
| Stimulus-drive term | 100 | 96 | 100 |
| Target-locked gain | 73 | 65 | 90 |
| Sustained gain | 83 | 77 | 87 |
| Distractor-locked gain | 23 | 27 | 50 |
| Small location-specific term A | 20 | 35 | 80 |
| Small location-specific term B | 7 | 12 | 73 |

(Second column preliminary, 26 of 30 subjects.) All numbers on the same voxels.

## A real effect can lose cross-validation

Adding a term with true value θ and per-subject estimation variance σ²_θ changes the expected
held-out SSE by about `(σ²_θ − θ²)·ΣX²`. **If |θ| < its per-subject SE, the term hurts out of
sample even when it's real.** In the example, term A had θ ≈ 0.019 and fold SD ≈ 0.044, so it
had to lose, and its penalty showed up exactly where its regressor has its energy.

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
joint fit lets other parameters compensate, which inflates the term (median 2.9× across the 10
of 11 ROIs where the signs agreed, range 1.4–8×). Since `ΔSSE < 0` requires `|δ| < 2|b/a|`, an inflated δ overshoots and
*hurts*: the same term gave −12.6 × 10⁻⁵ in V2 with δ from the joint fit, +0.08 × 10⁻⁵ with
the conditional δ.

The result is a statement about the population, not about prediction for one subject's own
fit. Report it as such, with the effect size. In the example it was about 10⁻⁵ R².

## Time-series CV with moving stimuli (bars)

From a leave-one-run-out comparison on 3T bar data (CHN).

- **Pooled cvR² = 1 − SSE/SST** on the held-out run, with test data and predictions
  **demeaned per run**, so the training-mean null is 0 by construction. A time-shifted
  prediction is a weak null (it goes negative).
- **Report cvR² per test sweep direction.** When training and test bars sweep in opposite
  directions, position and HRF delay are confounded, and cvR² is negative for every method.
- **Concatenating training runs leaks**: the HRF tail of run 1 falls into run 2 (about 20 s
  of 610 TRs). Reset the HRF per run, or state the leak.
- **Select vertices on training runs only** (here: grid in-sample R² ≥ 0.1) and use the
  **same vertices for all models**.
- Free-HRF bounds get hit (13.9% of vertices at the delay bound): count them.
- **Check the aperture orientation** against an independent map (the dataset authors' maps):
  r(y, y0) must be positive (+0.99 here; r(y, −y0) = −0.99 is a flipped aperture).
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
