# Performance: hardware, backends, iterations

Measured in the benchmark for the braincoder paper ([Gilles86/paper_braincoder](https://github.com/Gilles86/paper_braincoder),
`notes/data/final/numbers_runtime.tsv`, regenerated 2026-11-16, Keras 3 with TF 2.20,
braincoder 0.7.1). Values are medians of 3 seeds (n noted where it is smaller), in **seconds
of internal fit time**. Each run is 2D Gaussian PRF + SPM HRF, with the fixed SPM kernel
(0.7.1; the 0.7.0 kernel was wrong, see `version_history.md`).

Datasets: synthetic grids of 490 / 4,900 / 49,000 voxels with T=200, and van Es 2019
(118,584 surface vertices, 93,534 non-NaN, which are the ones fitted, T=120). All final cells
share one protocol: NaN voxels dropped, `noise_model='ssq'`, `r2_atol=1e-4`, sigma grid from
1 degree, at most 10,000 iterations, fit to convergence on the synthetic grids too.

## Is a GPU worth it?

| Voxels | CPU 32 | A100 | Speedup |
|---|---:|---:|---:|
| 490 | 30 | 25 | 1.2× |
| 4,900 | 86 | 28 | 3.1× |
| 49,000 | 505 | 72 | 7.0× |
| 93,534 (real, to convergence) | 10,184 | 752 | 13.5× |

Full Gauss + GD, to convergence. On 49,000 voxels, CPU 8 / 16 / 32 took 1,107 / 717 / 505 s.
Real data: A100 TF 752 s (n = 3).

**Below a few thousand voxels a GPU buys almost nothing**, so don't queue for one. An
ROI fit is fine on CPU. Above that, take the GPU.

## Which GPU

Real dataset, TF backend:

| Hardware | Gaussian | DN |
|---|---:|---:|
| H200 | 434 | 980 |
| H100 | 484 | not run |
| A100 | 752 | 1,689 |
| L4 (24 GB) | 3,051 (51 min) | not run |

- Real data, 93,534 vertices, n = 3 seeds per cell. Ranges: A100 717 to 818, H100 473 to
  609, H200 384 to 436, L4 2,979 to 3,051. The L4 fits the whole surface at about 4× the A100 time.
- Treat differences of tens of percent between GPUs with caution (the spread across seeds
  and nodes is 10 to 30%).
- **The memory limit still matters for other models and backends.** For a static PRF, cost
  scales with voxels × timepoints × pixels, so smaller jobs fit more easily.
- The cluster's GPU inventory changes; see **sciencecluster**. V100s failed on TF 2.20
  (cuDNN) and have since been removed.
- For dynamic RF models (the RF moves every TR), the forward builds a per-TR Gaussian
  buffer and can reach about 20 GB at V=500, T≈3096, 2500 pixels. For an isotropic
  Gaussian on a rectangular grid, separating it into x and y factors with `einsum` is
  exact. In one project this took peak VRAM from over 24 GB to 642 MB and made it 13×
  faster. It helps little for static models.

## Which backend

- **Results are identical across backends and hardware**: real-data Gauss + GD R² agrees
  within 2e-5 (21 cells at 0.21830) and the synthetic `largegrid` within 3e-7 across TF/JAX,
  GPUs, CPU and seeds. The grid + Adam pipeline is deterministic, and so is the iteration
  count, so repeat seeds only to time it.
- **On GPU, the difference is small and goes either way**: real data, JAX is faster on A100
  (688 vs 752 s) and H200 (351 vs 434 s) and slower on H100 (582 vs 484 s, with a wide
  spread of 414 to 664). Use whichever env you have.
- **Torch out-of-memory in gradient descent is a braincoder leak, not allocator
  fragmentation.** In 0.7.0 and earlier `ParameterFitter.fit` computes R² between steps with
  autograd on and keeps the graph (about 5 voxel × pixel tensors per iteration), which ran 12
  cells (A100/H100/H200, 49,000 and 93,534 voxels) out of memory. `expandable_segments` was
  already on in those runs and does not help. Fixed in 0.7.1 (`fix/torch-gd-leak`); with the
  fix, real data took 330 s on an H100 and 275 s on an H200 with torch. Before 0.7.1, use TF
  or JAX for whole-surface fits or wrap the tracking in `torch.no_grad()`. Small ROI fits
  were fine.
- **One env per backend on GPU.** TF, JAX and torch CUDA stacks conflict.
- Laptop (Apple Silicon, one run, 500 voxels × 774 TR × 2500 pixels, s/iteration): torch
  on MPS 0.06, JAX on CPU 0.10, TF on CPU 0.19. Torch needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1` for `lstsq` and `solve`.

## How many iterations

Real dataset, A100, SSQ loss, default `r2_atol` (an earlier sweep with the 0.7.0 HRF kernel;
the shape of the curve is what to use, not the R² values):

| Max iterations | 10 | 100 | 300 | 1,000 | 3,000 | 10,000 |
|---|---:|---:|---:|---:|---:|---:|
| Mean R² | 0.149 | 0.155 | 0.182 | 0.211 | 0.216 | 0.219 |
| Median R² | 0.042 | 0.054 | 0.084 | 0.092 | 0.096 | 0.099 |
| Seconds | 50 | 79 | 140 | 357 | 964 | 3,108 |

So about 0.3 s/iteration here, and the gains are nearly exhausted by 1,000–3,000.

- **`r2_atol=1e-4` is the default since 0.7.** The previous default, 1e-6, was unreachable: improvement
  per 100 iterations plateaus around 1e-5, so it runs to the iteration cap. 1e-4 stops
  near iteration 3,000, about 3× faster, for a cost of under 0.001 in R².
- **Gradient descent matters on real data but not on clean synthetic data.** Grid + OLS
  to full fit: mean R² 0.149 → 0.216 on van Es, 0.280 → 0.275 on a synthetic grid with a
  denser search. A synthetic benchmark will understate what it does.
- **Loss:** `'ssq'` is the default. In a test on 100 synthetic voxels (T=240, 6,000 iterations):
  - From a good grid, `'gaussian'` matches `'ssq'` iteration by iteration and takes about 25% longer.
  - From a coarse grid, it leaves about 2–5× less R² gap at 3,000 iterations (e.g. 0.9 vs 4.6 × 10⁻⁴).
    The gain is there with equal noise across voxels too, and it comes with log-SSQ alone.
  - A sum-of-R² loss follows the SSQ path: Adam scales each parameter by its own gradient history, so a
    constant per-voxel rescaling changes almost nothing. It does not balance noisy voxels.
  - Earlier notes claimed about 5× fewer iterations (50 voxels, moment-matched start). Don't quote that
    as a general speedup.

## Extras and their cost

- **Flexible HRF can matter, and its cost depends on the stop rule.** `r2_atol` stops when
  the mean R² improves by less than `r2_atol` over 100 iterations, so models differ in
  **iteration count**, not in cost per iteration (real data, A100: Gauss + GD 3,154
  iterations, free HRF 2,632, DN 4,287; about 0.23 s per iteration for all three, the same on
  every seed and backend). The free HRF therefore ran *faster* than Gauss (608 vs 752 s)
  because it stopped earlier; quote time per iteration with the count and do not call a
  variant cheaper because it stopped sooner. At a fixed 100 iterations it is still about 5×
  more expensive per iteration on 49k voxels (an earlier measurement, 303 vs 58 s on A100).
  Real data R² 0.314 vs 0.218 (in-sample), and one participant-level HRF recovers most of it
  (SKILL.md, Timeseries PRFs).
- **DN** costs about 2.2× a Gaussian on the same GPU (1,689 vs 752 s on an A100, mostly more
  iterations).
- **Dense grid alone** (real data, A100) takes 145 s and already reaches an R² comparable to
  other packages' final fits (see Comparing packages); most of braincoder's time is Adam
  gradient descent under a tight `r2_atol`. A looser `r2_atol` trades a little R² for much time.
- **Grid search** barely scales with CPU threads: 49k voxels took 550 / 458 / 376 s on
  8 / 16 / 32 cores. For chunking, `ParameterFitter(memory_limit=…)` sets the chunk size;
  lower it on OOM.
- **Parameter recovery on synthetic data** (49k voxels): x, y recover about equally well for
  every package (median position error 0.64 to 0.86 degrees); `sd` is the weak one. With the
  corrected SPM kernel braincoder's `sd` Spearman ρ is 0.728 and bias −1.27 under the SPM
  HRF, 0.789 and −0.13 under the HRF that generated the data (popeye 0.720, mrVista 0.774).
  It is the HRF, not the package: see "Comparing packages". Do not read much into
  single-voxel `sd` with a mismatched HRF.

## Comparing packages

From the paper benchmark (`notes/analyses/competitor_verification.md` and the speed section
of `paper/draft.md` in paper_braincoder). Speed on the real data (93,534 vertices, A100
unless stated): GEM-pRF 57 s, prfmodel 202 to 293 s (JAX / TF), braincoder 752 s (TF), prfpy
491 s on 32 CPUs (one run), popeye 12,832 s on 32 CPUs. The gap is the optimizer and its stop
rule, not the stimulus or the grid (all read the same design matrix; braincoder's grid of
12,500 points equals prfpy's): braincoder runs Adam on all vertices until mean R² improves by
less than 1e-4 (3,154 iterations), prfmodel takes a fixed 1,000 steps, prfpy runs L-BFGS-B
per voxel, GEM-pRF refines without iterating. On a fixed set of 32,108 well-fitted vertices
the dense grid alone (145 s) reached in-sample R² 0.399, against 0.395 prfpy, 0.393 popeye,
0.369 GEM-pRF, 0.358 prfmodel; gradient descent raised braincoder to 0.411. Do not claim
braincoder is the fastest pRF fitter. GEM-pRF fits only the isotropic Gaussian with a fixed
HRF and has no decoding.

- **Compare on one fixed voxel set** and state it. popeye constrains amplitude to be positive
  and braincoder does not, so whole-brain mean R² is not comparable.
- **R² cannot detect a failed fit.** analyzePRF had the highest real-data R² (0.412) while 11%
  of its vertices had sigma > 30 degrees, and on synthetic data its sigma collapsed to
  0.07 degrees (ρ = −0.01). Always show sigma recovery.
- **Check the coordinate frame** by r(y, y0) against r(y, −y0): `prfanalyze-popeye:1.0.1`
  reverses y; prfpy and analyzePRF need a `flipud` of the aperture; prfmodel and GEM-pRF need
  none.
- **Sigma conventions differ**: GEM-pRF sigma can be negative (use |sigma|); analyzePRF sigma
  = rfsize × sqrt(expt) in pixels; 5.9% of prfmodel's sigmas sat at its 0.05 grid floor.
- **The HRF differs** (prfpy SPM + derivative; prfmodel two-gamma + derivative; AFNI and
  popeye canonical), and a wrong HRF shrinks sigma at large true sigma in every package. Match
  the HRF to the generator before ranking optimisers: mrVista's top R² came from data
  generated with its own HRF.
- **State HRF, stop rule, hardware and date with every timing**: real-data and synthetic
  timings come from different dates, nodes and stop rules, so do not quote speed-up multiples
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
PYTHON=$HOME/data/conda/envs/<env>/bin/python sbatch --account=<acct> --partition=<cpu-partition> \
    --cpus-per-task=16 scripts/bench_fit.sh
PYTHON=… sbatch --account=<acct> --partition=<gpu-partition> --gres=gpu:1 --constraint=<gpu> scripts/bench_fit.sh
```

`bench_fit.sh` requests `--mem=64G`, which is too little at 50,000 voxels (the paper's
benchmark jobs on it ran out of memory): pass `--mem=128G` for those rows. `bench_fit.py`
imports `tensorflow` directly, so it cannot time JAX or torch. `bench_fit.sh` does not set
the backend, so export `KERAS_BACKEND` yourself in a Keras 3 env, and record `braincoder.__version__` next to the results. **Only compare rows from the
same braincoder code path.** A pre-Keras TF fork compiles with `tf.function`, while the
Keras 3 port runs eagerly, so timings from one say little about the other. Timing also
varies by node, so run a few repeats before concluding that more CPUs made a job slower.

## The pre-Keras TF line (`tf.function`, `feature/f-beta-r2-mixture`)

Some projects pin this older code path, which compiles the training step, so **don't compare
these rows with the Keras 3 tables above.** Measured 2026-11-02 with `scripts/bench_fit.py`
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

A fit on real data, per fold, 8 CPUs: **400 voxels × 2300 TRs of a modulation-field model
took 28 min of Adam (90th percentile 58 min) for the per-voxel pRF stage, then 1.5–2 min of
L-BFGS per shared-term model.** Dynamic-field forwards cost far more per voxel than the
static rows above. Loading the BOLD adds 4 min per subject (about 270k mask voxels × 12 runs)
if you mask after loading the full brain, so cache ROI data when you loop over ROIs.
