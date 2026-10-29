#!/usr/bin/env bash
# Step 1 of docs/claims_plan.md: token log-probs (Qwen, OLMo) and greedy verbalized
# probabilities (Qwen) for every dice query. Needs one GPU with at least 24 GB (bf16).
#
# Compute nodes may have no internet. On the login node, first:
#   python -m pip install -e ".[lm]"
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
#   huggingface-cli download allenai/OLMo-2-1124-7B-Instruct --revision 470b1fba1ae01581f270116362ee4aa1b97f4c84
#   python -m robust_likelihood.dice build
#
# Replace every <PLACEHOLDER> before submitting.
#   Slurm: sbatch scripts/dice_job.sh
#   LSF:   bsub < scripts/dice_job.sh
#   PBS:   qsub scripts/dice_job.sh
# Each scheduler reads only its own directives, so one file serves all three.
#
#SBATCH --job-name=dice-lm
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=dice-lm.%j.log
#
#BSUB -J dice-lm
#BSUB -q <GPU_QUEUE>
#BSUB -n 4
#BSUB -gpu "num=1:gmem=24G"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 03:00
#BSUB -o dice-lm.%J.out
#BSUB -e dice-lm.%J.err
#
#PBS -N dice-lm
#PBS -q <GPU_QUEUE>
#PBS -l select=1:ncpus=4:ngpus=1:mem=48gb
#PBS -l walltime=03:00:00
#PBS -P <PROJECT_ID>
#PBS -j oe
#PBS -o dice-lm.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-${LS_SUBCWD:-$(pwd)}}}"

# Cluster environment: load modules and activate the environment the extras were installed into.
# module load <CUDA_MODULE>
# source <VENV_DIR>/bin/activate
export HF_HOME="${HF_HOME:-<HF_CACHE_DIR>}"
export HF_HUB_OFFLINE=1

nvidia-smi

# Each model writes its own run folder under results/dice/; the manifest records the
# exact revision loaded and the self-checks (tokenization boundary, exact greedy decoding).
python -m robust_likelihood.dice score --local --model qwen --batch-size 32
python -m robust_likelihood.dice score --local --model olmo --batch-size 32
