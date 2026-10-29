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
# Replace every <PLACEHOLDER> before submitting, or override it on the command line.
#   Slurm: sbatch scripts/dice_job.sh
#   NUS SoC: sbatch --ignore-pbs -p gpu --gres=gpu:h100-47:1 --time=02:30:00 \
#            --export=ALL,VENV_DIR=/var/tmp/cuizq-rl,VENV_TAR=$HOME/envs/cuizq-rl.tar scripts/dice_job.sh
#   (--ignore-pbs: sbatch would otherwise read the #PBS and #BSUB lines too; the GPU type
#   must be named, a100-40, a100-80, h100-47, or h100-96, since a feature constraint with
#   an untyped GPU request is refused)
#   (home is NFS, so the environment is built once on local disk and packed; VENV_TAR is
#   unpacked onto the compute node's local disk when VENV_DIR is not there yet. Login-node
#   /tmp has a 50 MB quota, hence /var/tmp.)
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

# Cluster environment: load modules if the cluster needs them, and set VENV_DIR to the
# environment the extras were installed into.
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

# Each model writes its own run folder under results/dice/; the manifest records the
# exact revision loaded and the self-checks (tokenization boundary, exact greedy decoding).
python -m robust_likelihood.dice score --local --model qwen --batch-size 32
python -m robust_likelihood.dice score --local --model olmo --batch-size 32
