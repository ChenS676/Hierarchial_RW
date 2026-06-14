#!/bin/bash
PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_mamba/bin/python
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

NUM_LAYERS=(2 3)
HIDDEN_DIMS=(128 256)

cd $DIR
for NL in "${NUM_LAYERS[@]}"; do
  for HD in "${HIDDEN_DIMS[@]}"; do
    # use lower lr for larger hidden dim
    if [ "$HD" -eq 256 ]; then LR=0.001; else LR=0.005; fi

    echo "=== num_layers=$NL  hidden_dim=$HD  lr=$LR ==="
    $PYTHON run_ddi.py \
      --num_layers $NL \
      --hidden_dim $HD \
      --lr $LR \
      --walk_length 16 \
      --window_size 4 \
      --sample_rate 0.5 \
      --num_epochs 100 \
      --patience 30 \
      --global_batch_size 65536 \
      --walk_encoder conv \
      --dropout 0.3 \
      --wb_project neuralwalker-ddi-arch-search \
      --exp_name nr${NR}_nl${NL}_hd${HD}
  done
done
