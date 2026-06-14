#!/bin/bash
PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_mamba/bin/python
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

WALK_LENGTHS=(8 16 32 50)
WINDOW_SIZES=(4 8 16)
SAMPLE_RATES=(0.5 1.0)

cd $DIR
for WL in "${WALK_LENGTHS[@]}"; do
  for WS in "${WINDOW_SIZES[@]}"; do
    for SR in "${SAMPLE_RATES[@]}"; do
      echo "=== walk_length=$WL  window_size=$WS  sample_rate=$SR ==="
      $PYTHON run_ddi.py \
        --walk_length $WL \
        --window_size $WS \
        --sample_rate $SR \
        --num_epochs 100 \
        --patience 30 \
        --lr 0.005 \
        --global_batch_size 65536 \
        --hidden_dim 128 \
        --num_layers 2 \
        --walk_encoder conv \
        --dropout 0.3 \
        --wb_project neuralwalker-ddi-walk-search \
        --exp_name wl${WL}_ws${WS}_sr${SR}
    done
  done
done
