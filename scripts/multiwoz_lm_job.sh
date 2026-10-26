#!/usr/bin/env bash
# Phase 8 of docs/multiwoz_size_principle_plan.md: language-model surprisal and
# normalized typicality for every probe row. Needs one GPU with at least 24 GB (bf16).
# Do not submit until the user has approved the GPU job.
#
# Replace every <PLACEHOLDER> before submitting.
#   LSF: bsub < scripts/multiwoz_lm_job.sh
#   PBS: qsub scripts/multiwoz_lm_job.sh
# Each scheduler reads only its own directives, so one file serves both.
#
#BSUB -J multiwoz-lm
#BSUB -q <GPU_QUEUE>
#BSUB -n 4
#BSUB -gpu "num=1:gmem=24G"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 03:00
#BSUB -o multiwoz-lm.%J.out
#BSUB -e multiwoz-lm.%J.err
#
#PBS -N multiwoz-lm
#PBS -q <GPU_QUEUE>
#PBS -l select=1:ncpus=4:ngpus=1:mem=48gb
#PBS -l walltime=03:00:00
#PBS -P <PROJECT_ID>
#PBS -j oe
#PBS -o multiwoz-lm.log

set -euo pipefail

cd "${PBS_O_WORKDIR:-${LS_SUBCWD:-$(pwd)}}"

# Cluster environment: load modules and activate an environment with Python 3.11 or later.
# module load <CUDA_MODULE>
# source <VENV_DIR>/bin/activate
export HF_HOME="${HF_HOME:-<HF_CACHE_DIR>}"

python -m pip install --quiet -e ".[lm]"
nvidia-smi

# The manifest records the exact Hugging Face revision hash that was loaded.
# Pass --revision <HASH> to pin it on a rerun.
python -m robust_likelihood.multiwoz lm-score \
    --probe data/multiwoz/derived/probe.jsonl \
    --batch-size 16 \
    --label lm
