#!/bin/bash
#SBATCH --job-name=abl_rs
#SBATCH --output=logs/abl_rs_%j.out
#SBATCH --error=logs/abl_rs_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --partition=p.hpcl93

mkdir -p logs

cd /fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/Plaintoid

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralwalker
uv sync

echo "=============================================="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Ablation: Recurrent Steps"
echo "Start: $(date)"
echo "=============================================="

nvidia-smi

SEEDS="2025 2026 1010 1012"
DATA_NAME="Cora"
DATA_ROOT="./data/Cora"
RUN_TAG="ablation_walk_hierarchy"

run_ablation() {
    local exp_name=$1
    local rs=$2
    local wl=$3
    local nw=$4
    local bs=$5
    local device_idx=$6
    local start_core=$7
    local k=$8

    local end_core=$((start_core + k - 1))
    echo "GPU $device_idx | cores $start_core-$end_core | $exp_name | rs=$rs wl=$wl nw=$nw bs=$bs | $(date)"

    CUDA_VISIBLE_DEVICES=$device_idx \
    taskset -c $start_core-$end_core \
    uv run ablations/link_prediction_ablation.py \
        --data_name  $DATA_NAME \
        --data_root  $DATA_ROOT \
        --recurrent_steps $rs \
        --walk_length     $wl \
        --num_walks       $nw \
        --global_batch_size $bs \
        --run_tag  $RUN_TAG \
        --exp_name $exp_name \
        --seeds    $SEEDS

    echo "Done: $exp_name (exit $?) | $(date)"
}

# GPU 0 — Recurrent Steps rs=1, rs=2  (fixed budget wl*nw=128/rs)
(
    run_ablation ExpRS1 1 16 8 256  0  0 10
    run_ablation ExpRS2 2  8 8  16  0  0 10
) &

# GPU 1 — Recurrent Steps rs=4, rs=8
(
    run_ablation ExpRS4 4  4 8 16  1 10 10
    run_ablation ExpRS8 8  2 8 16  1 10 10
) &

wait
echo "=============================================="
echo "Recurrent Steps done. End: $(date)"
echo "=============================================="
