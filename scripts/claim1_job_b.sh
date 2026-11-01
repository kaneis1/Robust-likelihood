#!/usr/bin/env bash
# Step 2d of docs/claims_plan.md, GPU job B: greedy verbalized probabilities from Qwen on the
# robustness subset, after recovering the broad|expl paraphrases the extractor used to reject.
# Needs one GPU with at least 24 GB (bf16). Do not submit until the user has approved it.
#
# One stage, within a 3-hour GPU slot:
#   1. rewrites-recheck  re-checks the full rewrite run's regex rejections with the current
#                        extractor (NLI on the GPU, no API calls) and rewrites
#                        data/multiwoz/derived/rewrites.jsonl with the recovered rewrites
#   2. claim1-lm         scores only the recovered rewrites, as job A did: Qwen hrewrite and
#                        hsample, OLMo hrewrite
#   3. verbalcheck local Qwen, greedy, exact and meaning framings plus normalization
#
# Needs, besides the repo: the full rewrite run folder (results/ is not in git) and a finished
# Phase 8 run in results/multiwoz. On the login node first, with the environment's Python:
#   huggingface-cli download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
#       --revision b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7
#   (Qwen and OLMo are already there from job A; the NLI tokenizer needs sentencepiece)
#
# Replace every <PLACEHOLDER> before submitting, or override it on the command line.
#   Slurm: sbatch scripts/claim1_job_b.sh
#   LSF:   bsub < scripts/claim1_job_b.sh
#   PBS:   qsub scripts/claim1_job_b.sh
#   NUS SoC: sbatch --ignore-pbs -p gpu --gres=gpu:h100-47:1 --time=03:00:00 \
#            --export=ALL,VENV_DIR=/var/tmp/cuizq-rl,VENV_TAR=$HOME/envs/cuizq-rl.tar scripts/claim1_job_b.sh
#
#SBATCH --job-name=claim1-verbal
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=claim1-verbal.%j.log
#
#BSUB -J claim1-verbal
#BSUB -q <GPU_QUEUE>
#BSUB -n 4
#BSUB -gpu "num=1:gmem=24G"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 03:00
#BSUB -o claim1-verbal.%J.out
#BSUB -e claim1-verbal.%J.err
#
#PBS -N claim1-verbal
#PBS -q <GPU_QUEUE>
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=03:00:00
#PBS -P <PROJECT_ID>
#PBS -j oe
#PBS -o claim1-verbal.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-${LS_SUBCWD:-$(pwd)}}}"
FULL_REWRITES="${FULL_REWRITES:-results/multiwoz/rewrites/20261029T103256Z-full}"

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

newest() {
    for dir in $(ls -1d $1 2>/dev/null | sort -r); do
        if grep -q finished_at "$dir/manifest.json"; then echo "$dir"; return; fi
    done
    echo "no finished run matching $1" >&2
    exit 1
}

test -f "$FULL_REWRITES/candidates.jsonl" || { echo "copy $FULL_REWRITES to the cluster first" >&2; exit 1; }
PHASE8="$(newest 'results/multiwoz/*-phase8')"

python -m robust_likelihood.multiwoz rewrites-recheck "$FULL_REWRITES" --nli-device cuda
RECHECK="$(newest 'results/multiwoz/rewrites/*-recheck')"
python -m robust_likelihood.multiwoz claim1-lm --model qwen --parts hrewrite,hsample --phase8 "$PHASE8" \
    --units-from "$RECHECK" --batch-size 64 --label recheck-qwen
python -m robust_likelihood.multiwoz claim1-lm --model olmo --parts hrewrite --phase8 "$PHASE8" \
    --units-from "$RECHECK" --batch-size 64 --label recheck-olmo
python -m robust_likelihood.multiwoz verbalcheck local --model qwen --phase8 "$PHASE8" --label verbal-qwen
