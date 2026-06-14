#!/bin/bash
PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_mamba/bin/python
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

cd $DIR
for SEED in 2025 2026 2027 1010 1012; do
  echo "=== seed=$SEED ==="
  $PYTHON run_ddi.py \
    --seed $SEED \
    --num_epochs 200 \
    --patience 50 \
    --walk_length 16 \
    --window_size 4 \
    --sample_rate 0.5 \
    --hidden_dim 128 \
    --num_layers 2 \
    --lr 0.005 \
    --global_batch_size 65536 \
    --walk_encoder conv \
    --dropout 0.3 \
    --wb_project neuralwalker-ddi-final \
    --exp_name seed${SEED}
done
