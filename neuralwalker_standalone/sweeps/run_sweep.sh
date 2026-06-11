#!/usr/bin/env bash
# Launch a W&B grid sweep over lr and global_batch_size (16 runs total).
#
# Usage:
#   bash run_sweep.sh                  # create sweep + run 1 agent here
#   bash run_sweep.sh <SWEEP_ID>       # attach an agent to an existing sweep

set -euo pipefail

cd "$(dirname "$0")"

ENV=nw_mamba          # micromamba env name
SWEEP_CFG=sweep_lr_bs.yaml
ENTITY=""             # set to your W&B entity/team, or leave empty

# ── 1. Create sweep (skip if SWEEP_ID already provided) ──────────────────────
if [[ $# -ge 1 ]]; then
    SWEEP_ID="$1"
    echo "Attaching to existing sweep: $SWEEP_ID"
else
    if [[ -n "$ENTITY" ]]; then
        SWEEP_ID=$(micromamba run -n "$ENV" wandb sweep --entity "$ENTITY" "$SWEEP_CFG" 2>&1 \
                   | grep -oP '(?<=sweep ID: )\S+')
    else
        SWEEP_ID=$(micromamba run -n "$ENV" wandb sweep "$SWEEP_CFG" 2>&1 \
                   | grep -oP '(?<=sweep ID: )\S+')
    fi
    echo "Created sweep: $SWEEP_ID"
fi

# ── 2. Run agent (runs all assigned configs sequentially) ────────────────────
echo "Starting sweep agent …"
micromamba run -n "$ENV" wandb agent "$SWEEP_ID"
