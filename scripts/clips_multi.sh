#!/usr/bin/env bash
# Several CLIPS replays in one CPU job, each in its own Julia process (single-threaded, 5-7 GB each).
# The SoC cluster runs at most 16 jobs per user, so packing replays of plan chunks into one job is how
# they run in parallel. From the repository root:
#   sbatch -p long -c 4 --mem=40G -t 24:00:00 scripts/clips_multi.sh mdkg "qwen3_8b 1.1.keys,2.1.keys" "osim_mid 3.2.keys"
# Each argument after the domain is "<model> <comma-separated plans>"; each replay logs to
# clips-replay.<job>.<model>.<first plan>.log and writes to results/clips/<model>/<domain>/plans-<first plan>/.
#
#SBATCH --job-name=clips-multi
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=clips-multi.%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
DOMAIN="${1:?pass mdkg or vh}"
shift
pids=()
for spec in "$@"; do
    read -r model plans <<< "$spec"
    bash scripts/clips_job.sh replay "$DOMAIN" "$model" "$plans" \
        > "clips-replay.${SLURM_JOB_ID:-local}.${model}.${plans%%,*}.log" 2>&1 &
    pids+=("$!")
    echo "started $model $plans (pid $!)"
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
echo "finished: $failed of ${#pids[@]} replays failed"
exit "$failed"
