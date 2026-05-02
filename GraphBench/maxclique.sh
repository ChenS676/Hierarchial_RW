#!/bin/bash

run_models() {
    local DATASET=$1
    local MODEL=$2
    local device_idx=$3

    local SEEDS="2025 2026 2027"
    local batch_size=256
    local n_hidden=256
    local lr=1e-4
    local epochs=10
    local train_subset_ratio=0.1

    local EXTRA=""
    case $MODEL in
        hrw) EXTRA="--layers 1 --num_heads 8 --num_walks 10 --walk_length 8" ;;
        gin) EXTRA="--layers 6" ;;
        gt)  EXTRA="--layers 4 --num_heads 8" ;;
    esac

    for SEED in $SEEDS; do
        echo "[$(date +%H:%M:%S)] $MODEL | $DATASET | seed=$SEED | GPU=$device_idx"
        uv run maxclique_${MODEL}.py \
            --dataset_name       ${DATASET} \
            --seed               ${SEED} \
            --hidden_dim         ${n_hidden} \
            --batch_size         ${batch_size} \
            --epochs             ${epochs} \
            --adam_max_lr        ${lr} \
            --train_subset_ratio ${train_subset_ratio} \
            --use_pe             false \
            --eval_metric_class  algoreas_classification \
            ${EXTRA} \
            >> logs/${MODEL^^}_${DATASET}_s${SEED}.out 2>&1
    done
}

mkdir -p logs

DATASETS=("maxclique_easy" "maxclique_medium" "maxclique_hard")

for DATASET in "${DATASETS[@]}"; do
    echo "=== Launching: $DATASET ==="
    run_models $DATASET hrw 0 &
    run_models $DATASET gin 0 &
    run_models $DATASET gt  0 &
done

# Wait and report
FAILED=0
for JOB in $(jobs -p); do
    wait $JOB || FAILED=$((FAILED + 1))
done

[ $FAILED -eq 0 ] && echo "✓ All done." || echo "✗ ${FAILED} jobs failed."

echo "── F1 summary ──"
grep -h "Best Test F1" logs/*.out 2>/dev/null | sort