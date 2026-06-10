#!/bin/bash
#SBATCH --job-name=abl_wl_nw
#SBATCH --output=logs/abl_wl_nw_%j.out
#SBATCH --error=logs/abl_wl_nw_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=p.hpcl93

mkdir -p logs

cd /fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/Plaintoid

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralwalker
uv sync 

echo "=============================================="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Ablation: Walk Length + Num Walks"
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

# GPU 0 — Walk Length  (nw=4, rs=1)
(
    run_ablation ExpB_WalkLength 1  4  4 256  0  0 10
    run_ablation ExpB_WalkLength 1  8  4 256  0  0 10
    run_ablation ExpB_WalkLength 1 16  4 256  0  0 10
    run_ablation ExpB_WalkLength 1 32  4 128  0  0 10
) &

# GPU 1 — Num Walks  (wl=8, rs=1)
(
    run_ablation ExpA_numberWalks 1 8  2 256  1 10 10
    run_ablation ExpA_numberWalks 1 8  4 256  1 10 10
    run_ablation ExpA_numberWalks 1 8  8 256  1 10 10
    run_ablation ExpA_numberWalks 1 8 16 256  1 10 10
) &

wait
echo "=============================================="
echo "Walk Length + Num Walks done. End: $(date)"
echo "=============================================="
