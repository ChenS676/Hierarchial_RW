#!/bin/bash
# Run 5 seeds with the best config found from sweep.
# Edit the hyperparameters below before running.

PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_cu124/bin/python
SCRIPT=run_ddi.py
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

# ── Best config (fill in after sweep) ────────────────────────────────────────
HIDDEN_DIM=128
NUM_LAYERS=2
WALK_LENGTH=16
WALK_ENCODER=conv
DROPOUT=0.3
LR=0.005
# ─────────────────────────────────────────────────────────────────────────────

for SEED in 2025 2026 2027 2028 2029; do
  echo "=== Seed $SEED ==="
  cd $DIR && $PYTHON $SCRIPT \
    --seed $SEED \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --walk_length $WALK_LENGTH \
    --walk_encoder $WALK_ENCODER \
    --dropout $DROPOUT \
    --lr $LR \
    --num_epochs 500 \
    --patience 50 \
    --wb_project neuralwalker-ddi-multiseed \
    --exp_name seed${SEED}
done
