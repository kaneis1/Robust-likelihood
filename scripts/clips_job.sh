#!/usr/bin/env bash
# CLIPS with small local language models (docs/clips_small_lm_plan.md): the Julia side, on a CPU node.
#
#   record <domain> [plans]   run julia/clips/<domain>_experiments.jl in record mode; every language-model
#                             query goes to results/clips/record/<domain>.jsonl (plans: comma-separated ids)
#   replay <domain> <model>   run it in replay mode with results/clips/tables/<model>.jsonl; CLIPS's
#                             result files go to results/clips/<model>/<domain>/
#
# Needs the environment packed by scripts/clips_setup.sh ($CLIPS_TAR). From the repository root:
#   sbatch -p normal -c 4 --mem=16G -t 03:00:00 scripts/clips_job.sh record mdkg
#
#SBATCH --job-name=clips
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=clips.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

STAGE="${1:?pass record or replay}"
DOMAIN="${2:?pass mdkg or vh}"
JULIA_VERSION="${JULIA_VERSION:-1.10.7}"
CLIPS_LOCAL="${CLIPS_LOCAL:-/var/tmp/$USER-clips}"
CLIPS_TAR="${CLIPS_TAR:-$HOME/envs/clips-julia.tar}"

(
    flock 9
    if [ ! -x "$CLIPS_LOCAL/julia-$JULIA_VERSION/bin/julia" ]; then
        tar -xf "$CLIPS_TAR" -C "$(dirname "$CLIPS_LOCAL")"
    fi
) 9>"$(dirname "$CLIPS_LOCAL")/.$(basename "$CLIPS_LOCAL").lock"
export JULIA_DEPOT_PATH="$CLIPS_LOCAL/depot"
export JULIA_PKG_OFFLINE=true
JULIA="$CLIPS_LOCAL/julia-$JULIA_VERSION/bin/julia"
test -z "$(git -C external/CLIPS.jl status --porcelain)"

if [ "$STAGE" = "record" ]; then
    export CLIPS_MODE=record
    export CLIPS_RECORD_LOG="results/clips/record/$DOMAIN${3:+-smoke}.jsonl"
    export CLIPS_PARTICLE_LOG="${CLIPS_RECORD_LOG%.jsonl}.particles.jsonl"
    export CLIPS_PLANS="${3:-}"
    rm -f "$CLIPS_RECORD_LOG" "$CLIPS_PARTICLE_LOG"
    /usr/bin/time -v "$JULIA" --project=julia/clips "julia/clips/${DOMAIN}_experiments.jl"
    wc -l "$CLIPS_RECORD_LOG"
elif [ "$STAGE" = "replay" ]; then
    MODEL="${3:?pass the model key}"
    export CLIPS_MODE=replay
    export CLIPS_TABLE="results/clips/tables/$MODEL.jsonl"  # python -m robust_likelihood.clips table --model $MODEL
    # A subset of plans (comma-separated) runs in its own subfolder, named after its first plan, so chunks of
    # one model started in the same second never write the same timestamped file; the evaluation merges them.
    export CLIPS_PLANS="${4:-}"
    export CLIPS_RESULTS="results/clips/$MODEL/$DOMAIN${4:+/plans-${4%%,*}}"
    "$JULIA" --project=julia/clips "julia/clips/${DOMAIN}_experiments.jl"
else
    echo "unknown stage $STAGE" >&2
    exit 2
fi
