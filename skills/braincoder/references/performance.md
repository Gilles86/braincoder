# Performance: hardware, backends, iterations

Rough, qualitative guidance from the maintainers' own runs (Keras 3, braincoder 0.7.1, 2D
Gaussian PRF + SPM HRF). Your hardware and data will differ: **time your own setup on a
subset before sizing a large job** (see "Measuring your own env" below).

## Is a GPU worth it?

- **Below a few thousand voxels a GPU buys almost nothing**, so don't queue for one. An ROI
  fit is fine on CPU.
- **For whole-brain or whole-surface fits (tens of thousands of voxels and up), use a GPU.**
  The speed-up grows with the number of voxels; a whole cortical surface fits in minutes on a
  recent data-centre GPU and takes hours on a CPU node.
- Treat differences of tens of percent between GPU types with caution: the spread across
  seeds and nodes is of that order.
- Cost scales with voxels × timepoints × pixels; chunk with `ParameterFitter(memory_limit=…)`
  if you run out of GPU memory.
- For dynamic RF models (the RF moves every TR), the forward pass builds a per-TR Gaussian
  buffer that can reach tens of GB. For an isotropic Gaussian on a rectangular grid,
  separating it into x and y factors with `einsum` is exact and cuts peak memory and runtime
  dramatically.

## Which backend

- **Results are the same across backends and hardware** (up to float32 rounding): grid +
  Adam is deterministic, and so is the iteration count, so repeat seeds only to time it.
- **On GPU, TF and JAX are similar in speed**, and which is faster depends on the GPU. Use
  whichever env you have.
- **Torch out-of-memory in gradient descent before 0.7.1 is a braincoder leak**, not
  allocator fragmentation: `ParameterFitter.fit` computed R² between steps with autograd on
  and kept the graph, so large fits ran out of GPU memory (`expandable_segments` does not
  help). Fixed in 0.7.1. Before 0.7.1, use TF or JAX for large fits, or wrap the tracking in
  `torch.no_grad()`.
- **One env per backend on GPU.** TF, JAX and torch CUDA stacks conflict.
- On Apple Silicon laptops, torch on MPS is the fastest backend; it needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1` for `lstsq` and `solve`.

## How many iterations

- **`r2_atol=1e-4` is the default since 0.7.** It stops when the mean R² improves by less
  than 1e-4 over 100 iterations, typically after a few thousand iterations, at a negligible
  cost in R². The previous default, 1e-6, was effectively unreachable and ran to the cap.
- **Most of the gain comes in the first 1,000–3,000 Adam iterations**; beyond that, time
  grows much faster than R².
- **Gradient descent matters on real data more than on clean synthetic data.** A synthetic
  benchmark with a dense grid will understate what it does.
- **Loss:** `'ssq'` is the default. `'gaussian'` follows it closely from a good grid and is
  a little slower; from a coarse grid it can close the remaining R² gap faster. A sum-of-R²
  loss does not balance noisy voxels (Adam rescales each parameter anyway).

## Extras and their cost

- **Compare models by time per iteration together with the iteration count.** With the
  `r2_atol` stop rule, models differ mainly in how many iterations they run, so a richer model
  can finish sooner; do not call a variant cheaper because it stopped earlier.
- **Flexible HRF** costs several times more per iteration than a fixed HRF. Whether it is
  worth its parameters is a question for cross-validation; try one participant-level HRF
  first (SKILL.md, Timeseries PRFs).
- **DN** takes roughly twice as long as a Gaussian, mostly through more iterations.
- **Grid search** barely scales with CPU threads. Most of the total time is Adam under a
  tight `r2_atol`; a looser `r2_atol` trades a little R² for much time.
- **Parameter recovery on synthetic data:** position (x, y) recovers well; `sd` is the weak
  parameter and is sensitive to the HRF used in the fit. Do not read much into single-voxel
  `sd` when the HRF may be wrong.

## Comparing packages

If you benchmark braincoder against another PRF package:

- **Compare on one fixed voxel set** and state it. Packages differ in constraints (for
  example, whether amplitude must be positive), so whole-brain mean R² is not comparable.
- **R² cannot detect a failed fit.** A package can reach a high R² while many of its sigmas
  sit at absurd values or at a grid floor. Always show sigma recovery on synthetic data.
- **Check the coordinate frame** by r(y, y0) against r(y, −y0): packages differ in whether
  the aperture needs a vertical flip.
- **Sigma conventions differ** between packages (sign, units, pixel versus degree); convert
  before comparing.
- **The HRF differs** between packages, and a wrong HRF shrinks sigma at large true sigma
  in every package. Match the HRF to the generator before ranking optimisers, and be wary
  of synthetic data generated with one package's own HRF.
- **The optimizer and its stop rule dominate speed**, not the stimulus or the grid. State
  HRF, stop rule, hardware and date with every timing, and do not quote speed-up multiples
  without them.

## Scaling a CPU job

Pin threads to the allocation, or `cpu8` and `cpu32` jobs behave alike:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK TF_NUM_INTEROP_THREADS=2
```

## Measuring your own env

```bash
PYTHON=/path/to/env/bin/python BENCH_PY=/path/to/scripts/bench_fit.py \
    sbatch --account=<acct> --partition=<cpu-partition> --cpus-per-task=16 scripts/bench_fit.sh
PYTHON=… BENCH_PY=… sbatch --account=<acct> --partition=<gpu-partition> --gres=gpu:1 --constraint=<gpu> scripts/bench_fit.sh
```

`bench_fit.sh` requests `--mem=64G`, which is too little at 50,000 voxels: pass `--mem=128G`
for those rows. `bench_fit.py` imports `tensorflow` directly, so it cannot time JAX or torch.
`bench_fit.sh` does not set the backend, so export `KERAS_BACKEND` yourself in a Keras 3 env,
and record `braincoder.__version__` next to the results. **Only compare rows from the
same braincoder code path.** A pre-Keras TF fork compiles with `tf.function`, while the
Keras 3 port runs eagerly, so timings from one say little about the other. Timing also
varies by node, so run a few repeats before concluding that more CPUs made a job slower.

## The pre-Keras TF line (`tf.function`, `feature/f-beta-r2-mixture`)

Some projects pin this older code path, which compiles the training step, so **don't compare
these rows with Keras 3 timings.** Measured 2026-11-02 with `scripts/bench_fit.py`
(random binary paradigm, 2500 pixels, TF 2.14), one run each.

Seconds per Adam iteration, Gaussian with fixed HRF:

| Voxels | T | CPU 8 | CPU 16 (slow node) | CPU 32 |
|---:|---:|---:|---:|---:|
| 500 | 774 | 0.020 | 0.11 | 0.018 |
| 2,000 | 774 | 0.062 | 0.25 | 0.035 |
| 10,000 | 774 | 0.23 | 1.32 | 0.15 |
| 50,000 | 774 | 0.89 | 7.04 | 0.74 |
| 500 | 3,096 | 0.041 | 0.22 | 0.025 |
| 2,000 | 3,096 | 0.15 | 0.85 | 0.079 |
| 10,000 | 3,096 | 0.62 | not reached | 0.32 |
| 50,000 | 3,096 | 2.7 | not reached | 1.7 |

- **The node matters more than the core count.** 8 → 32 cores gained 1.1–2×, while one
  16-core job on another node was 4–8× slower than the 8-core job. I don't know why (load or
  CPU type); rerun on another node before concluding anything from a slow job.
- **Cost is about linear in voxels and in T**, and DoG adds only about 20% per iteration
  (0.27 vs 0.23 s at 10,000 voxels, T=774, CPU 8; flexible HRF and DN rows are still pending).
- **Memory:** DoG with V=50,000 × T=3,096 was killed at `--mem=64G` on both CPU jobs that
  reached it; the same size Gaussian passed. Split voxels into chunks well below that.

On real data, dynamic-field (modulation-field) forwards cost far more per voxel than the
static rows above: budget tens of minutes of Adam per fold for a few hundred voxels and a
few thousand TRs on a handful of CPUs, and time a small subset first. Loading the BOLD can
add minutes per subject if you mask after loading the full brain, so cache ROI data when you
loop over ROIs.
