# Convergence: is it the model, or the optimizer?

The advice below comes from fitting nested ladders of models with shared terms on top of
per-voxel pRFs. Check it on your own data.

## Check monotonicity on pooled SSE, never on mean per-voxel R²

A nested model must not fit its *training* data worse than its parent. If it does, its
ΔcvR² is optimizer noise. But **test this on total SSE**. The loss sums squared error over
voxels, so it can fall while the unweighted mean of per-voxel R² also falls.

- **Mean R² raises false alarms.** A nested fit that is monotonic by construction (L-BFGS
  from the parent's optimum, below) can still look "worse" on mean per-voxel R² in a
  substantial share of folds. On total SSE, or per-voxel R² weighted by each voxel's training
  SST, the same fits are never worse than the parent.
- **Independent refits really are noisy**, but less than mean R² suggests, and most of the
  non-monotonicity sits in low-signal regions.
- A drop in training fit from optimizer noise can be larger than the ΔcvR² effects you are
  trying to compare, so check before you interpret a small ΔcvR².

To avoid needing weights afterwards, store training SSE and SST per voxel.

## Fit nested models from the parent

| Procedure | What happens |
|---|---|
| Refit all per-voxel pRF parameters per model | Training fit can be non-monotonic (above); shared gains vary widely across subjects |
| Freeze pRFs at the parent's fold fit; fit only shared terms from the parent's optimum, new terms at their null value | Monotonic by construction; much smaller cross-subject spread of the shared terms |

The parent start reproduces the parent's prediction exactly. L-BFGS only accepts descent,
so the richer model cannot end worse.

## Box the shared parameters

Unbounded shared σ and gain terms run away in directions the data do not constrain: σ below
the size of a stimulus item, or a σ so wide that the field is flat and trades off against
amplitude. Without bounds, a sizeable share of folds ended at absurd gains or widths; with a
box (for example σ 0.5–8°, |gain| ≤ 10) such folds became rare.

Medians hardly move, but means and spreads are useless without the box. Check for values
sitting *on* a bound: they indicate an unidentified direction, not a finding.

## L-BFGS on shared terms

- **Compute the SSE in float64.** In float32 the summation noise (about 1e-7 relative) stops
  L-BFGS early with `ABNORMAL` line searches. The prediction may stay float32.
- **Restart from the end point** (fresh Hessian) until SSE stops improving; a couple of
  rounds is typical. An `ABNORMAL` status after a restart usually means there's nothing left
  to gain.
- **Several σ starts** (the parent's, plus a spread such as 0.5, 1, 2, 4°) cost proportionally
  more than a single start. They usually agree; they are insurance for the fits that don't.
- **Time:** L-BFGS on a handful of shared terms is fast compared with Adam on all per-voxel
  pRF parameters (see `performance.md`).

## A final all-free pass ("polish") is rarely worth it

L-BFGS over all parameters from the clamped optimum is expensive (folds can hit the
iteration cap and the job's walltime), changes cvR² by about nothing, and has side effects:
lower training SSE, less stable pRF `sd`, and shared gains that shrink as the pRFs absorb them.

## Per-voxel parameter fits with Adam

- Spatial and HRF parameters go in **separate stages**.
- Grid → refine → Adam; see `SKILL.md`. For stopping rules and iteration counts see
  `performance.md`.
- A plateau on `r2_atol` is not proof of convergence. If a nested check matters, use it.
