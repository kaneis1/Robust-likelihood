#!/usr/bin/env bash
# CLIPS with small local language models (docs/clips_small_lm_plan.md), Phase 0 setup. CPU only.
#
#   1. clones probcomp/CLIPS.jl at the pinned commit into external/CLIPS.jl (never edited);
#   2. installs Julia (JULIA_VERSION) and instantiates julia/clips/Project.toml, CLIPS.jl's own pins,
#      with InversePlanning from cosilab/InversePlanning.jl#release-0.0.1; julia/clips/Manifest.toml
#      records the resolved versions;
#   3. packs Julia and its depot into $CLIPS_TAR, which later jobs unpack onto their node's local disk
#      (the home directory is NFS, slow for many small files, and the login node caps memory at 1 GB,
#      too little to precompile);
#   4. downloads the paper's OSF project (https://osf.io/v8ru7/: datasets, results) into data/clips/raw/.
#
# Run on a compute node with internet access, from the repository root:
#   sbatch -p normal -c 8 --mem=32G -t 02:00:00 --wrap "bash scripts/clips_setup.sh"
#
#SBATCH --job-name=clips-setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=clips-setup.%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CLIPS_COMMIT=6536a2a86d6b4837481a00e630a2e302c90a5278
JULIA_VERSION="${JULIA_VERSION:-1.10.7}"
CLIPS_LOCAL="${CLIPS_LOCAL:-/var/tmp/$USER-clips}"
CLIPS_TAR="${CLIPS_TAR:-$HOME/envs/clips-julia.tar}"

# 1. CLIPS.jl at the pinned commit.
if [ ! -d external/CLIPS.jl/.git ]; then
    git clone https://github.com/probcomp/CLIPS.jl external/CLIPS.jl
fi
git -C external/CLIPS.jl fetch -q origin
git -C external/CLIPS.jl checkout -q "$CLIPS_COMMIT"
test "$(git -C external/CLIPS.jl rev-parse HEAD)" = "$CLIPS_COMMIT"
test -z "$(git -C external/CLIPS.jl status --porcelain)"

# 2. Julia and the environment, on the node's local disk.
mkdir -p "$CLIPS_LOCAL"
if [ ! -x "$CLIPS_LOCAL/julia-$JULIA_VERSION/bin/julia" ]; then
    curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-$JULIA_VERSION-linux-x86_64.tar.gz" \
        | tar -xz -C "$CLIPS_LOCAL"
fi
export JULIA_DEPOT_PATH="$CLIPS_LOCAL/depot"
export JULIA_PKG_PRECOMPILE_AUTO=0
JULIA="$CLIPS_LOCAL/julia-$JULIA_VERSION/bin/julia"
"$JULIA" --version
"$JULIA" --project=julia/clips -e '
    using Pkg
    Pkg.add(url="https://github.com/cosilab/InversePlanning.jl", rev="release-0.0.1")
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status()
    using PDDL, SymbolicPlanners, Gen, GenParticleFilters, InversePlanning, GenGPT3, CSV, DataFrames
    println("CLIPS environment loads")'

# 3. Pack it for later jobs.
mkdir -p "$(dirname "$CLIPS_TAR")"
tar -cf "$CLIPS_TAR" -C "$(dirname "$CLIPS_LOCAL")" "$(basename "$CLIPS_LOCAL")"
ls -la "$CLIPS_TAR"

# 4. The paper's OSF project, folder by folder through the OSF API.
python3 - <<'PY'
import json, os, urllib.request

def listing(url):
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        yield from page["data"]
        url = page["links"].get("next")

def fetch(url, path):
    for entry in listing(url):
        attributes = entry["attributes"]
        target = os.path.join(path, attributes["name"])
        if attributes["kind"] == "folder":
            fetch(entry["relationships"]["files"]["links"]["related"]["href"], target)
        else:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(target) or os.path.getsize(target) != attributes["size"]:
                urllib.request.urlretrieve(entry["links"]["download"], target)
            print(target, attributes["size"])

root = "https://api.osf.io/v2/nodes/v8ru7/files/osfstorage/"
for entry in listing(root):
    if entry["attributes"]["kind"] == "folder" and entry["attributes"]["name"] in ("datasets", "results"):
        fetch(entry["relationships"]["files"]["links"]["related"]["href"],
              os.path.join("data/clips/raw", entry["attributes"]["name"]))
PY
echo "setup finished"
