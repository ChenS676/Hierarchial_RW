#!/bin/bash
PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_mamba/bin/python
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

LRS=(0.001 0.003 0.005)
BATCH_SIZES=($((2**16)) $((2**17)) $((2**18)) $((2**19)) $((2**20)))

cd $DIR
for LR in "${LRS[@]}"; do
  for BS in "${BATCH_SIZES[@]}"; do
    echo "=== lr=$LR  batch_size=$BS ==="
    $PYTHON run_ddi.py \
      --lr $LR \
      --global_batch_size $BS \
      --num_epochs 20 \
      --patience 30 \
      --hidden_dim 128 \
      --num_layers 2 \
      --walk_length 16 \
      --walk_encoder conv \
      --dropout 0.3 \
      --wb_project neuralwalker-ddi-sweep \
      --exp_name lr${LR}_bs${BS}
  done
done
