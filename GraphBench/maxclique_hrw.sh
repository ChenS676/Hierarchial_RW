#!/bin/bash
# run_hrw_maxclique.sh

mkdir -p logs

SEEDS="2025 2026 2027"
BATCH_SIZE=512
HIDDEN=256
LR=2e-4
EPOCHS=300
SUBSET=0.2

run_hrw() {
    local DATASET=$1

    for SEED in $SEEDS; do
        echo "[$(date +%H:%M:%S)] START | $DATASET | seed=$SEED"

        uv run hrw_maxcliques.py \
            --dataset_name       ${DATASET} \
            --seed               ${SEED} \
            --hidden_dim         ${HIDDEN} \
            --encoding_dim       ${HIDDEN} \
            --layers             1 \
            --num_heads          8 \
            --batch_size         ${BATCH_SIZE} \
            --test_batch_size    128 \
            --epochs             ${EPOCHS} \
            --adam_max_lr        ${LR} \
            --train_subset_ratio ${SUBSET} \
            --walk_length        8 \
            --num_walks          10 \
            --node2vec_p         1.0 \
            --node2vec_q         1.0 \
            --recurrent_steps    1 \
            --ffn_multiplier     4.0 \
            --grad_clip_norm     0.5 \
            --dropout            0.1 \
            --use_nw_pe          false \
            --use_lap_pe         false \
            --use_rwse           false \
            --wandb_project      bench_maxclique \
            --log_every          10 \
            --eval_metric_class  algoreas_classification \
            >> logs/HRW_${DATASET}_s${SEED}.out 2>&1

        echo "[$(date +%H:%M:%S)] DONE  | $DATASET | seed=$SEED"
    done
}

# 3 datasets in parallel — SLURM handles CPU affinity automatically
# run_hrw maxclique_easy   
# run_hrw maxclique_medium 
run_hrw maxclique_hard   

FAILED=0
for JOB in $(jobs -p); do
    wait $JOB || FAILED=$((FAILED + 1))
done

echo ""
[ $FAILED -eq 0 ] && echo "✓ All done." || echo "✗ ${FAILED} failed."
echo ""
echo "── Final F1 summary ──────────────────────"
grep -h "Final →" logs/HRW_*.out 2>/dev/null | sort