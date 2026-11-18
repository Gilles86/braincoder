#!/bin/bash -l
#SBATCH --job-name=braincoder_bench
#SBATCH --output=/dev/null
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Benchmark braincoder fitting on CPU or GPU. Pass account/partition/resources
# on the command line, e.g.
#   PYTHON=/path/to/env/bin/python BENCH_PY=/path/to/bench_fit.py \
#       sbatch --account=<acct> --partition=<cpu-partition> --cpus-per-task=16 bench_fit.sh
#   PYTHON=... BENCH_PY=... sbatch --account=<acct> --partition=<gpu-partition> --gres=gpu:1 \
#       --constraint=<gpu> --cpus-per-task=4 bench_fit.sh
set -eo pipefail
OUT_DIR="${OUT_DIR:-$HOME/logs/braincoder_bench}"
mkdir -p "$OUT_DIR"
TAG="$(hostname -s)_${SLURM_JOB_ID}"
exec >"$OUT_DIR/${TAG}.log" 2>&1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export TF_NUM_INTEROP_THREADS=2
fi
[[ -n "${CUDA_LIB_DIR:-}" ]] && export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# sbatch runs a spooled copy, so locate bench_fit.py explicitly.
BENCH_PY="${BENCH_PY:?set BENCH_PY to the absolute path of bench_fit.py}"
"${PYTHON:?set PYTHON to an env with braincoder + TF}" -u "$BENCH_PY" \
    --out "$OUT_DIR/${TAG}.tsv" ${BENCH_ARGS}
