# Convergence: is it the model, or the optimizer?

Examples come from a nested ladder of attention-field (modulation-field) models on a 7T
dataset: 30 subjects × 11 ROIs × 4 folds, up to 500 voxels per ROI, about 2300 training
TRs, 2500 stimulus pixels. Code was the pre-Keras TF line. The patterns generalise; the
absolute numbers won't.

## Check monotonicity on pooled SSE, never on mean per-voxel R²

A nested model must not fit its *training* data worse than its parent. If it does, its
ΔcvR² is optimizer noise. But **test this on total SSE**. The loss sums squared error over
voxels, so it can fall while the unweighted mean of per-voxel R² also falls.

Share of folds where the richer model is below its parent on training data (13,200 folds;
"variance-weighted" = per-voxel R² weighted by that voxel's training SST):

| Term added | Independent refits: mean R² | Independent refits: variance-weighted | Nested fit: mean R² | Nested fit: variance-weighted |
|---|---:|---:|---:|---:|
| Target gain | 19% | 3% | 16% | **0%** |
| Distractor gain | 32% | 5% | 25% | **0%** |
| Sustained gain | 21% | 16% | 20% | **0%** |
| Target + distractor | 28% | 11% | 26% | **0%** |
| Target + distractor + sustained | 16% | 14% | 23% | **0%** |
| Location-specific sustained | 46% | 23% | 33% | **0%** |
| Location-specific distractor | 29% | 14% | 26% | **0%** |
| Both location-specific | 32% | 13% | 29% | **0%** |

Read it this way:

- **Mean R² raises a false alarm.** The nested fit is monotonic by construction (L-BFGS SSE
  ended above the parent's start in 0 of 10,384 fits), yet mean R² "worsens" in up to a
  third of folds.
- **Independent refits really are noisy**, but less than mean R² suggests: 3–23% of folds,
  with most of it in low-signal regions (m7-vs-m4: 3–7% in V1–V3, 19–27% in IPS, FEF, SPL1).
- In one fold, the richer model's variance-weighted R² dropped by as much as 0.01. That is
  far larger than the 1e-3 ΔcvR² effects it was being compared with.

The weights are approximate where the stored SST is demeaned within condition. To avoid
needing them, store training SSE and SST per voxel.

## Fit nested models from the parent

| Procedure | Result on the same data |
|---|---|
| Refit all ~3.5k per-voxel pRF parameters per model | Train-fit non-monotonic (above); cross-subject IQR of shared gains up to 1.16 |
| Freeze pRFs at the parent's fold fit; fit only shared terms from the parent's optimum, new terms at their null value | Monotonic by construction; IQR falls 1.7–19× (target gain 0.69 → 0.32, sustained 1.16 → 0.06, location-specific distractor 0.44 → 0.26) |

The parent start reproduces the parent's prediction exactly. L-BFGS only accepts descent,
so the richer model cannot end worse.

## Box the shared parameters

Unbounded shared σ and gain terms run away in directions the data do not constrain: σ below
the size of a stimulus item, or a σ so wide that the field is flat and trades off against
amplitude.

Same fits, models with shared terms (a fold counts as extreme at |gain| ≥ 10 or σ outside
0.1–50°; the bounded maximum sits exactly on that gain threshold, so bounded rows are
upper limits):

| Fit | Folds extreme | Fits with an extreme fold | Largest \|gain\| | σ range |
|---|---:|---:|---:|---|
| Unbounded | 23% | 44% | 1.4 × 10⁶ | 0.0–150,000° |
| Bounded (σ 0.5–8°, \|gain\| ≤ 10, item scale ≤ 50) | ≤ 0.5% | ≤ 1.5% | 10 | 0.5–8° |

Medians hardly move, but means and spreads are useless without the box. Check for values
sitting *on* a bound: they indicate an unidentified direction, not a finding.

## L-BFGS on shared terms

- **Compute the SSE in float64.** In float32 the summation noise (about 1e-7 relative) stops
  L-BFGS early with `ABNORMAL` line searches. The prediction may stay float32.
- **Restart from the end point** (fresh Hessian) until SSE stops improving; 2 rounds was
  typical. An `ABNORMAL` status after a restart usually means there's nothing left to gain.
- **Several σ starts** (parent's, 0.5, 1, 2, 4°) cost 5× a single start. In the fit I
  inspected they agreed to 7 digits; they are insurance for the fits that don't.
- **Time:** per fold, about 400 voxels on 8 CPUs, L-BFGS on the shared terms took 1.5–2 min
  with 5 starts; Adam on all per-voxel pRF parameters took about 28 min (see `performance.md`).

## A final all-free pass ("polish") is rarely worth it

L-BFGS over all parameters from the clamped optimum, 500 voxels, one subject × ROI:

| | |
|---|---|
| Runtime | 12 h walltime hit part-way through the ladder; folds reach the 9,000-iteration cap |
| ΔcvR² polished − clamped | +0.0002 to +0.001, about 0 for the base models |
| Side effects | Lower training SSE, less stable pRF `sd`, gains shrink as pRFs absorb them |

## Per-voxel parameter fits with Adam

- Spatial and HRF parameters go in **separate stages**.
- Grid → refine → Adam; see `SKILL.md`. For stopping rules and iteration counts see
  `performance.md`.
- A plateau on `r2_atol` is not proof of convergence. If a nested check matters, use it.
