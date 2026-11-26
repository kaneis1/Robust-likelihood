#!/usr/bin/env bash
# Simulator study (docs/simulator_study_plan.md) on one GPU (at least 24 GB, bf16). Do not submit
# until the user has approved it.
#
#   check   Phase 1: every model and format of configs/simulator.json, 8 samples for each of 6
#           seeded goals, boundary and exact-sampling checks, P0 against Phase 8. About 5 min.
#   score   Phase 2, one model and format per job: sim_job.sh score <model> <format> [parts]
#           (parts comma-separated; default all the plan runs for the model). About 0.5 H100-hour,
#           1-1.5 with hsample (osim_mid, osim_final, qwen25_inst).
#   pilot   Phase 5, one model per job: sim_job.sh pilot <model>. User turns 2 and 3 of the 60
#           dialogues of configs/simulator.json phase5, 32 samples per (dialogue, turn, hypothesis).
#           About 0.5-1 H100-hour.
#   noise   Noise floor of the utterance-rewrite shifts: sim_job.sh noise <model:format,...>. Each
#           Phase 2 urewrite part again with the original utterances, and the human part again.
#           About 4 min per model.
#   week    docs/week_nov24_plan.md: sim_job.sh week <model:format,...> <parts> [sim-score arguments],
#           one sim-score run per model in turn, parts of uperturb, xscore, hsample (e.g. xscore needs
#           --samples-dir <the model's Phase 2 run>; a smoke run adds --first-items N --hsample-limit N).
#   student docs/week_nov24_plan.md Part D: sim_job.sh student <R-lin,R-mlp,C-mlp> <scheme> <phase2> <xscore run>
#           [student-train arguments], one training run per name in turn, each with its predictions.
#   clips   docs/clips_small_lm_plan.md Phase 2: sim_job.sh clips <model,...> (keys of configs/clips.json),
#           per model in turn the scores of data/clips/derived/pairs.jsonl, then its replay table in
#           results/clips/tables/. About 5-10 minutes per model.
#   clips-perturbed  Phase 4: sim_job.sh clips-perturbed <model,...>, per model in turn the co-batched
#           scores of data/clips/derived/batches.jsonl (python -m robust_likelihood.clips perturb first).
#
# On the login node first, every model at its pinned revision (configs/simulator.json), with
# huggingface_hub; the jobs run with HF_HUB_OFFLINE=1 and never pip install.
#
# Replace every <PLACEHOLDER> before submitting, or override it on the command line.
#   Slurm: sbatch scripts/sim_job.sh <stage>
#   LSF:   bsub < scripts/sim_job.sh   (set STAGE in the environment)
#   PBS:   qsub -v STAGE=<stage> scripts/sim_job.sh
#   NUS SoC: sbatch --ignore-pbs -p gpu --gres=gpu:h100-47:1 --time=03:00:00 \
#            --export=ALL,VENV_DIR=/var/tmp/cuizq-rl,VENV_TAR=$HOME/envs/cuizq-rl.tar scripts/sim_job.sh check
#            (the same with: scripts/sim_job.sh score osim_mid S)
#
#SBATCH --job-name=sim-study
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sim-study.%j.log
#
#BSUB -J sim-study
#BSUB -q <GPU_QUEUE>
#BSUB -n 4
#BSUB -gpu "num=1:gmem=24G"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 03:00
#BSUB -o sim-study.%J.out
#BSUB -e sim-study.%J.err
#
#PBS -N sim-study
#PBS -q <GPU_QUEUE>
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=03:00:00
#PBS -P <PROJECT_ID>
#PBS -j oe
#PBS -o sim-study.log

set -euo pipefail

STAGE="${1:-${STAGE:?pass the stage: check, score, pilot, noise, week, student, clips or clips-perturbed}}"
cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-${LS_SUBCWD:-$(pwd)}}}"

# module load <CUDA_MODULE>
if [ -n "${VENV_TAR:-}" ]; then
    # Jobs sharing a node take turns, so none starts on a half-unpacked environment.
    (
        flock 9
        if [ ! -x "${VENV_DIR:?VENV_TAR needs VENV_DIR}/bin/python" ]; then
            df -h "$(dirname "$VENV_DIR")"
            tar -xf "$VENV_TAR" -C /
        fi
    ) 9>"$(dirname "$VENV_DIR")/.$(basename "$VENV_DIR").lock"
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

if [ "$STAGE" = "check" ]; then
    python -m robust_likelihood.multiwoz sim-check --all --phase8 "$(newest_phase8)"
elif [ "$STAGE" = "score" ]; then
    MODEL="${2:-${MODEL:?pass the model key}}"
    FORMAT="${3:-${FORMAT:?pass the format}}"
    PARTS="${4:-${PARTS:-}}"
    python -m robust_likelihood.multiwoz sim-score --model "$MODEL" --format "$FORMAT" ${PARTS:+--parts "$PARTS"}
elif [ "$STAGE" = "pilot" ]; then
    MODEL="${2:-${MODEL:?pass the model key}}"
    python -m robust_likelihood.multiwoz sim-pilot --model "$MODEL"
elif [ "$STAGE" = "noise" ]; then
    RUNS="${2:-${RUNS:?pass model:format,...}}"
    python -m robust_likelihood.multiwoz sim-noise --runs "$RUNS"
elif [ "$STAGE" = "week" ]; then
    RUNS="${2:-${RUNS:?pass model:format,...}}"
    PARTS="${3:-${PARTS:?pass the parts}}"
    shift 3 || true
    for RUN in ${RUNS//,/ }; do
        python -m robust_likelihood.multiwoz sim-score --model "${RUN%%:*}" --format "${RUN##*:}" --parts "$PARTS" "$@"
    done
elif [ "$STAGE" = "student" ]; then
    NAMES="${2:?pass the run names}"
    SCHEME="${3:?pass the bin scheme}"
    PHASE2="${4:?pass the Phase 2 run}"
    XSCORE="${5:?pass the xscore run, or newest}"
    shift 5
    if [ "$XSCORE" = "newest" ]; then
        # The newest finished, non-smoke sim-week run that has xscore.jsonl.
        XSCORE=""
        for dir in $(ls -1d results/multiwoz/*-simwk-qwen3_base-F 2>/dev/null | sort -r); do
            if [ -f "$dir/xscore.jsonl" ] && grep -q finished_at "$dir/manifest.json"; then XSCORE="$dir"; break; fi
        done
        if [ -z "$XSCORE" ]; then echo "no finished xscore run in results/multiwoz" >&2; exit 1; fi
        echo "xscore run: $XSCORE"
    fi
    for NAME in ${NAMES//,/ }; do
        python -m robust_likelihood.multiwoz student-train --run "$NAME" --scheme "$SCHEME" --phase2 "$PHASE2" --xscore "$XSCORE" "$@"
    done
elif [ "$STAGE" = "clips" ]; then
    MODELS="${2:?pass the model keys}"
    for MODEL in ${MODELS//,/ }; do
        python -m robust_likelihood.clips score --model "$MODEL"
        python -m robust_likelihood.clips table --model "$MODEL"
    done
elif [ "$STAGE" = "clips-perturbed" ]; then
    MODELS="${2:?pass the model keys}"
    for MODEL in ${MODELS//,/ }; do
        python -m robust_likelihood.clips score-perturbed --model "$MODEL"
    done
else
    echo "unknown stage $STAGE" >&2
    exit 2
fi
