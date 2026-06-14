#!/bin/bash
PYTHON=/fs/gpfs41/lv11/fileset01/pool/pool-shao/micromamba_root/envs/nw_mamba/bin/python
DIR=/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/neuralwalker_standalone

cd $DIR
for SEED in 2025 2026 2027 1010 1012; do
  echo "=== seed=$SEED ==="
  $PYTHON run_demo.py \
    --seed $SEED \
    --data_name PubMed \
    --wb_project neuralwalker-pubmed \
    --exp_name seed${SEED}
done
