#!/usr/bin/env bash
# Step 2c of docs/claims_plan.md, GPU job A: language-model scores for claim 1. Needs one GPU
# with at least 24 GB (bf16). Do not submit until the user has approved it.
#
# Two stages, each within a 3-hour GPU slot:
#   phase8   Phase 8 lm-score on the full probe (K = 64 per text) and the centering part
#   rewrites the rewrite parts with Qwen (urewrite, hrewrite, hsample), then OLMo scoring
# The rewrites stage needs data/multiwoz/derived/rewrites.jsonl and robustness_subset.json,
# and reads the newest finished Phase 8 run.
#
# Replace every <PLACEHOLDER> before submitting, or override it on the command line.
#   Slurm: sbatch scripts/claim1_job_a.sh <stage>
#   LSF:   bsub < scripts/claim1_job_a.sh   (set STAGE in the environment)
#   PBS:   qsub -v STAGE=<stage> scripts/claim1_job_a.sh
#   NUS SoC: sbatch --ignore-pbs -p gpu --gres=gpu:h100-47:1 --time=03:00:00 \
#            --export=ALL,VENV_DIR=/var/tmp/cuizq-rl,VENV_TAR=$HOME/envs/cuizq-rl.tar scripts/claim1_job_a.sh <stage>
#
# On the login node first: models downloaded with huggingface-cli (see scripts/dice_job.sh).
#
#SBATCH --job-name=claim1-lm
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=claim1-lm.%j.log
#
#BSUB -J claim1-lm
#BSUB -q <GPU_QUEUE>
#BSUB -n 4
#BSUB -gpu "num=1:gmem=24G"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 03:00
#BSUB -o claim1-lm.%J.out
#BSUB -e claim1-lm.%J.err
#
#PBS -N claim1-lm
#PBS -q <GPU_QUEUE>
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=03:00:00
#PBS -P <PROJECT_ID>
#PBS -j oe
#PBS -o claim1-lm.log

set -euo pipefail

STAGE="${1:-${STAGE:?pass the stage: phase8 or rewrites}}"
cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-${LS_SUBCWD:-$(pwd)}}}"

# module load <CUDA_MODULE>
if [ -n "${VENV_TAR:-}" ] && [ ! -x "${VENV_DIR:?VENV_TAR needs VENV_DIR}/bin/python" ]; then
    df -h "$(dirname "$VENV_DIR")"
    tar -xf "$VENV_TAR" -C /
fi
if [ -n "${VENV_DIR:-}" ]; then source "$VENV_DIR/bin/activate"; fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"

newest_phase8() {
    for dir in $(ls -1d results/multiwoz/*-phase8 2>/dev/null | sort -r); do
        if grep -q finished_at "$dir/manifest.json"; then echo "$dir"; return; fi
    done
    echo "no finished Phase 8 run in results/multiwoz" >&2
    exit 1
}

if [ "$STAGE" = "phase8" ]; then
    python -m robust_likelihood.multiwoz lm-score \
        --probe data/multiwoz/derived/probe.jsonl \
        --revision a09a35458c702b33eeacc393d103063234e8bc28 \
        --batch-size 32 --label phase8
    python -m robust_likelihood.multiwoz claim1-lm --model qwen --parts center --phase8 "$(newest_phase8)" \
        --batch-size 64 --label center-qwen
elif [ "$STAGE" = "rewrites" ]; then
    PHASE8="$(newest_phase8)"
    python -m robust_likelihood.multiwoz claim1-lm --model qwen --parts urewrite,hrewrite,hsample \
        --phase8 "$PHASE8" --batch-size 64 --label rewrites-qwen
    python -m robust_likelihood.multiwoz claim1-lm --model olmo --parts urewrite,hrewrite \
        --phase8 "$PHASE8" --batch-size 64 --label rewrites-olmo
else
    echo "unknown stage $STAGE" >&2
    exit 2
fi
