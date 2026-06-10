#!/bin/bash
# Submit all three ablation batch jobs across different partitions.
# Usage: bash ablations/submit_ablations.sh

cd /fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/Plaintoid

sbatch ablations/run_ablations_wl_nw.sbatch   # Walk Length + Num Walks  → p.hpcl93  (L40S)
sbatch ablations/run_ablations_rs.sbatch       # Recurrent Steps          → p.hpcl94g
sbatch ablations/run_ablations_budget.sbatch   # Budget (rs=1 vs rs=2)    → p.hpcl8*

echo "3 jobs submitted. Check status with: squeue -u $USER"
