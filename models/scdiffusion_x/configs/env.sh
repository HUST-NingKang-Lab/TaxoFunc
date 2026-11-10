#!/bin/bash
# Shared environment/paths for scDiffusion-X Phase-1 runs.
# Source this file from other scripts:  source "$(dirname "$0")/env.sh"

# Resolve the repository root from this file's location, so the code is portable.
_REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)"
export SCX_ROOT=${SCX_ROOT:-$_REPO_ROOT/models/scdiffusion_x}
export SCX_REPO=$SCX_ROOT/upstream

# wandb must not try to reach the network (offline only, or disabled)
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR=$SCX_ROOT/runs/wandb

export PYTHONPATH=$SCX_REPO/scdiffusionX/src:${PYTHONPATH:-}

# Activate the Python environment containing the required packages before
# sourcing this file (see requirements.txt).
