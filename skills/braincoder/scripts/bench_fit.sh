#!/bin/bash -l
#SBATCH --job-name=braincoder_bench
#SBATCH --output=/dev/null
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Benchmark braincoder fitting on CPU or GPU. Pass account/partition/resources
# on the command line, e.g.
#   PYTHON=$HOME/data/conda/envs/<env>/bin/python sbatch --account=<acct> --partition=standard \
#       --cpus-per-task=16 bench_fit.sh
#   PYTHON=... sbatch --account=<acct> --partition=lowprio --gres=gpu:1 --constraint=L4 \
#       --cpus-per-task=4 bench_fit.sh
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
BENCH_PY="${BENCH_PY:-$HOME/git/ruffgroup-claude-skills/skills/braincoder/scripts/bench_fit.py}"
"${PYTHON:?set PYTHON to an env with braincoder + TF}" -u "$BENCH_PY" \
    --out "$OUT_DIR/${TAG}.tsv" ${BENCH_ARGS}
