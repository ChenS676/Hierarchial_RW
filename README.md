# Hierarchical Random Walk (HierarchialRW)

A Transformer-based graph encoder that captures structure via **anonymized random walks** instead of message passing. Node neighbourhoods are encoded by sampling Node2Vec-style walks, anonymizing node IDs to relative first-occurrence indices, and processing the resulting sequences through a standard Transformer with Rotary Positional Embeddings (RoPE).

## Repository layout

```
Hierarchial_RW/
├── Plaintoid/          # Link prediction on Planetoid graphs (Cora, CiteSeer, PubMed)
│   ├── model.py            model architecture
│   ├── train.py            HeART training (standard)
│   ├── train_opt.py        HeART training with per-epoch profiling (CPU/GPU)
│   └── train_hrw_ablation.py  ablation: walk aggregation strategies
│
├── CityNet/            # Node classification on city road networks
│   ├── train_hrw.py        HierarchialRW + GNN baselines on CityNetwork
│   ├── hr_hrw_EAT.py       eccentricity / EAT analysis
│   ├── evalutors.py        evaluation helpers
│   └── requirements.txt
│
├── GraphBench/         # Algorithmic graph tasks (MST, MaxClique, MaxMatching, Bridges)
│   ├── hrw_mst.py
│   ├── hrw_maxcliques.py
│   ├── hrw_bridges_optimized_new.py
│   └── hrw_maxmatching.py  (train_matching_gin.py / train_matching_gt.py for baselines)
│
├── ablation_study_paper/  # HRW vs plain RW ablation (EAT, ECC, bottleneck, coverage)
│   ├── eval_hrw_eat_bottleneck.py
│   ├── hr_hrw_EAT.py
│   └── op_eval_hrw_vs_rw.py
│
├── lrgb/               # Long Range Graph Benchmark (GraphGPS-based baselines)
│   └── main.py
│
├── ogb/                # OGB link-prediction experiments
└── install_uv.sh       # one-shot environment bootstrap (uv)
```

## Model overview

| Component | Description |
|---|---|
| Walk sampling | Node2Vec (parameters `p`, `q`), `num_walks` walks of length `walk_length` per node |
| Anonymization | Replaces absolute node IDs with relative first-occurrence indices — structure-only, ID-agnostic |
| RoPE | Applied at anonymized positions, not raw sequence positions |
| Transformer | Pre-RMSNorm, SwiGLU FFN, stochastic depth (drop-path) |
| muP init | Maximal Update Parametrization for stable scaling |
| Optimizer | Muon (Nesterov + orthogonalization) for matrix weights; Adam for gains, biases, and MLP |
| Hierarchical variant | `DPCWUE` in `Plaintoid/train_opt.py` — bottom-up DP recurrence, memory O(N·W·L·D) |

## Environment setup

### Option A — uv (recommended)

```bash
# Install uv if needed
curl -Ls https://astral.sh/uv/install.sh | sh

# Plaintoid / ablation (Python 3.12, CUDA 12.8)
cd Plaintoid
uv sync

# CityNet (legacy, Python 3.11, CUDA 12.4)
cd CityNet
bash ../install_uv.sh          # creates .venv
pip install -r requirements.txt
```

### Option B — pip (CityNet / lrgb)

```bash
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install networkx osmnx wandb tqdm scikit-learn pandas muon-optimizer einops
```

## Main commands

### Link prediction — Plaintoid (HeART)

```bash
cd Plaintoid

# Basic run on Cora
uv run train.py --data_name Cora --data_root data/Cora

# CiteSeer with fixed splits
uv run train.py --data_name CiteSeer --data_root data/CiteSeer \
    --use_fixed_splits True --split_dir data/CiteSeer/fixed_splits

# Tune walk parameters
uv run train.py --data_name Cora --data_root data/Cora \
    --walk_length 16 --num_walks 32 --node2vec_p 1.0 --node2vec_q 0.5

# With Laplacian positional encodings
uv run train.py --data_name Cora --data_root data/Cora \
    --use_laplacian_pe True --laplacian_pe_path data/Cora/laplacian_pe.pt

# With DeepWalk feature augmentation
uv run train.py --data_name Cora --data_root data/Cora \
    --use_deepwalk_embeds True --deepwalk_pkl_path data/Cora/deepwalk.pkl
```

**Key arguments**

| Argument | Default | Description |
|---|---|---|
| `--data_name` | `Cora` | Dataset: `Cora`, `CiteSeer`, `PubMed` |
| `--num_epochs` | `300` | Training epochs |
| `--global_batch_size` | `256` | Positive-edge batch size |
| `--walk_length` | `8` | Walk length |
| `--num_walks` | `16` | Walks per node |
| `--node2vec_p/q` | `1.0` | Node2Vec return/in-out parameter |
| `--hidden_dim` | `128` | Transformer hidden dim |
| `--num_layers` | `1` | Transformer layers |
| `--num_heads` | `16` | Attention heads |
| `--recurrent_steps` | `1` | Hierarchical recurrence depth |
| `--muon_max_lr` | `1e-3` | Peak Muon learning rate |
| `--eval_metric` | `MRR` | Early-stopping metric (`MRR`, `AUC`, `Hits@K`) |
| `--patience` | `4` | Early-stopping patience (eval checks) |

### Link prediction — with profiling (`train_opt.py`)

```bash
cd Plaintoid
uv run train_opt.py --data_name Cora --data_root data/Cora \
    --recurrent_steps 3 --hidden_dim 256
# Saves per-epoch profiling → Cora/profiling/profiling_<run_id>.csv
# Appends run summary  → Cora/profiling/profiling_summary.csv
```

### Walk aggregation ablation

```bash
cd Plaintoid
# Default: flatten all walks into one long sequence
uv run train_hrw_ablation.py --data_name Cora --walk_agg flatten

# Alternative: process walks independently then concat
uv run train_hrw_ablation.py --data_name Cora --walk_agg concat
```

### Node classification — CityNet

```bash
cd CityNet

# GNN baseline
python train_hrw.py --method gcn   --dataset paris --device 0
python train_hrw.py --method sage  --dataset paris --device 0

# HierarchialRW
python train_hrw.py --method hierarchial_rw --dataset paris --device 0 \
    --hierarchial_rw_walk_length 8 \
    --hierarchial_rw_num_walks 16 \
    --hierarchial_rw_hidden_dim 128
```

### GraphBench — algorithmic tasks

```bash
cd GraphBench

# Minimum Spanning Tree
python hrw_mst.py

# Maximum Clique
python hrw_maxcliques.py

# Bipartite Maximum Matching (HRW)
python hrw_maxmatching.py

# Bridge detection
python hrw_bridges_optimized_new.py

# GIN / GT baselines
python train_matching_gin.py
python train_matching_gt.py
```

### Ablation study (HRW vs RW)

```bash
cd ablation_study_paper

# EAT + bottleneck evaluation
uv run eval_hrw_eat_bottleneck.py

# HRW vs plain RW on oracle metrics
uv run op_eval_hrw_vs_rw.py

# Single-metric plot
uv run single_plot.py
```

### Long Range Graph Benchmark (lrgb)

```bash
cd lrgb
conda activate lrgb

# GCN on Peptides-func
python main.py --cfg configs/GCN/peptides-func-GCN.yaml wandb.use False

# SAN on PascalVOC-SP
python main.py --cfg configs/SAN/vocsuperpixels-SAN.yaml wandb.use False
```

## Logging

All training scripts log to [Weights & Biases](https://wandb.ai). Set your entity/project with:

```bash
--wb_entity <your-entity> --wb_project <your-project>
```

To disable W&B:

```bash
export WANDB_MODE=offline   # or set inside the script
```

## Output files

| Path | Contents |
|---|---|
| `<dataset>/checkpoints/best_model_<run_id>.pth` | Best model checkpoint |
| `<dataset>/profiling/profiling_<run_id>.csv` | Per-epoch CPU/GPU profiling |
| `<dataset>/profiling/profiling_summary.csv` | Aggregated run summary |
| `<dataset>/experiment_results_link_prediction_updated_final.csv` | Per-run metrics |
