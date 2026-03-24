# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network (GNN)-based framework that translates phylogenetic structure into location-specific epidemiological parameters. It predicts per-location epidemiological metrics (R0, Recovery Rate, Source/Sink Score, Ancestral State) from BEAST2 phylogenetic trees.

## Overview

The repository contains three model pipelines and supporting infrastructure for data generation, batch processing, and evaluation:

| Pipeline | Description |
|----------|-------------|
| **stephy2** | Primary model using CBLV features + auxiliary tree statistics with custom edge-attention GAT (DTW-based) |
| **CBLV-CNN2** | Ablation baseline: CNN-only model with no graph structure (each node predicted independently) |
| **CBLV-GAT2** | Ablation baseline: standard GAT with node-based attention (no DTW edge features) |

## Project Structure

```
STEPHY/
├── simulate_and_extract/       # Data generation: BEAST2 simulation + feature extraction
│   ├── simulate_and_extract.sh # Main simulation loop
│   ├── submit.sh               # SLURM array job wrapper for HPC
│   ├── xml_generation.py       # Generate BEAST2/ReMaster XML configs
│   ├── characterizing_outbreak.py  # Extract labels from simulation outputs
│   ├── edit_tree.sh            # Strip unnecessary blocks from NEXUS files
│   └── simulation_engine.pdf   # ReMaster simulation engine documentation
├── simulate_and_extract_Denmark/  # Denmark-specific data generation
│   ├── simulate_and_extract.sh # Main simulation loop (Denmark settings)
│   ├── submit.sh               # SLURM array job wrapper for HPC
│   ├── xml_generation.py       # Generate XML configs with Denmark parameters
│   ├── characterizing_outbreak.py  # Extract labels from simulation outputs
│   ├── edit_tree.sh            # Strip unnecessary blocks from NEXUS files
│   └── r0_distribution_preview.png # Visualization of R0 Beta distribution
├── stephy2/                    # Primary model: CBLV-GAT with edge attention
│   ├── run_pipeline.sh         # End-to-end orchestration script
│   ├── analyze_trees.py        # Step 1: Determine num_locations and subtree_width
│   ├── build_graphs.py         # Step 2: Build DGL graphs
│   ├── train.py                # Step 3: Training loop
│   ├── model.py                # CBLV_GAT neural network
│   ├── data.py                 # Data loading, CBLV encoding, DTW, graph construction
│   ├── config.py               # Hyperparameters
│   └── beast2_parser.py        # Lightweight regex-based NEXUS parser
├── CBLV-CNN2/                  # Ablation: no graph structure
│   ├── model.py                # CBLV_CNN (wider CNN, no attention)
│   ├── train.py                # Training (no graph/edge logic)
│   └── ...                     # Same structure as stephy2
├── CBLV-GAT2/                  # Ablation: standard GATConv
│   ├── model.py                # CBLV_GAT using dgl.nn.GATConv
│   ├── train.py                # Training (adds self-loops, ignores edge features)
│   └── ...                     # Same structure as stephy2
├── batch_run/                  # HPC orchestration for large-scale runs
│   ├── README.md               # Step-by-step workflow instructions
│   ├── outbreak_check.py       # Data inspection and filtering
│   ├── build_graphs.py         # Batched graph building with filtering
│   ├── submit_build_graphs.sh  # SLURM job: parallel graph building
│   ├── merge_graphs.py         # Merge batch graph files into one
│   ├── submit_train.sh         # SLURM job: train all 12 models (3 pipelines x 4 labels)
│   ├── test.py                 # Evaluate all models on held-out test set
│   ├── beast2_parser.py        # NEXUS parser (shared from stephy2)
│   ├── config.py               # Hyperparameters (shared from stephy2)
│   └── data.py                 # Data loading (shared from stephy2)
└── README.md
```

## End-to-End Workflow

```
simulate_and_extract/                    simulate_and_extract_Denmark/
(generic, 10 locations)                  (5 Danish regions, real populations)
    │                                        │
    │  Generates: {id}_beast2.trees, {id}_nf.csv, {id}_parameter.csv
    └──────────────────┬─────────────────────┘
                       ▼
batch_run/outbreak_check.py ──► Determines num_locations & subtree_width
    │
    ▼
build_graphs.py (stephy2/ or batch_run/)
    │  Uses stephy2/data.py for CBLV + DTW + label extraction
    │  Produces: graphs.pt (DGL graphs with all node/edge features)
    ▼
    ├──► stephy2/train.py    (CBLV-GAT with edge attention)
    ├──► CBLV-CNN2/train.py  (CNN only, ignores edges)
    └──► CBLV-GAT2/train.py  (Standard GAT, ignores edge features, adds self-loops)
    │
    ▼
batch_run/test.py ──► Evaluates all 12 models on held-out test set
```

Graph building only needs to happen once -- `stephy2/build_graphs.py` produces a superset `graphs.pt` that all three pipelines consume.

## Model Architecture

### stephy2: CBLV-GAT with Edge Attention (Primary)

```
  CBLV (4, W) per node        Aux (5 tree stats) per node
        │                              │
  CBLVConvEncoder                 AuxBranch
    3 parallel Conv1d             Dense: 5 → 64 → 32
    - Plain: [12,24,48]                │
    - Stride: [12,24]                  │
    - Dilate: [12,24]                  │
        │                              │
    96-dim ─────────────────── 32-dim
                    │
              128-dim node embedding
                    │
           GraphEdgeAttention
             Edge features: DTW (distance, lag_mean, lag_std)
                    │
              256-dim (128 self + 128 aggregated)
                    │
           MLP: 256 → 128 → 64 → 32 → 1
                    │
           Per-node prediction
```

### CBLV-CNN2: CNN Baseline (Ablation)

- Wider CNN (192-dim) and wider AuxBranch (64-dim) to match 256-dim input
- No graph attention -- each node predicted independently
- Tests whether spatial aggregation helps

### CBLV-GAT2: Standard GAT (Ablation)

- Same CNN and AuxBranch as stephy2 (128-dim node embedding)
- Uses `dgl.nn.GATConv` (4 heads x 64 = 256-dim) instead of custom edge attention
- Fully connected graph with self-loops, no DTW edge features
- Tests whether DTW edge features provide value over standard node-based attention

## Features

**CBLV Encoding** (4 channels per node):
- Channel 0: Leaf distances from previous branch point
- Channel 1: Internal node depths (root distances)
- Channel 2: Accumulated edge lengths to leaves
- Channel 3: Accumulated edge lengths to internal nodes

**Auxiliary Tree Statistics** (5 per node): `mrca_depth`, `earliest_tip_time`, `latest_tip_time`, `avg_branch_length`, `n_tips`

**DTW Edge Features** (3 per edge, stephy2 only):
- DTW distance between KDE-smoothed tip-time curves
- Lag mean from DTW warping path
- Lag std from DTW warping path

**Normalization**:
- CBLV: divide by tree height
- Aux features: log-transform then z-score on training set
- Edge features: log+z-score for distance/lag_std; plain z-score for lag_mean
- Labels (regression): z-score; test predictions inverse-transformed

## Input Data Format

Each dataset folder requires paired files per outbreak:

```
input_folder/
  {id}_beast2.trees   # BEAST2 NEXUS file with annotated tips
  {id}_nf.csv         # Labels per location (one row per location)
```

**Tree tip annotation format:**
```
42[&type="I{3}",samp="sample",time=1.5]
     └── location=3, sampled tip, sampling time
```

**CSV columns:**

| Column | Description |
|--------|-------------|
| `Location` | Location index |
| `R0` | Basic reproduction number |
| `Recovery_Rate` | Recovery rate (mu) |
| `Source_Sink_Score` | (exports - imports) / (exports + imports), range [-1, 1] |
| `Ancestral_State` | One-hot: 1 if location is epidemic origin (MRCA) |

## Prediction Targets

| Target | Type | Metric |
|--------|------|--------|
| `R0` | Regression | R2, MSE |
| `Recovery_Rate` | Regression | R2, MSE |
| `Source_Sink_Score` | Regression | R2, MSE |
| `Ancestral_State` | Classification | Accuracy |

## Usage

### Data Generation

**Generic simulation engine** (randomized populations and locations):

```bash
# Generate N successful outbreaks (configure params via env vars in script)
bash simulate_and_extract/simulate_and_extract.sh <target_count> [output_folder]

# Run in parallel on HPC via SLURM
bash simulate_and_extract/submit.sh <num_batches> <sims_per_batch> <base_dir>
```

**Denmark simulation engine** (5 Danish regions with real populations):

```bash
# Same interface, Denmark-calibrated parameters
bash simulate_and_extract_Denmark/simulate_and_extract.sh <target_count> [output_folder]

# Run in parallel on HPC via SLURM
bash simulate_and_extract_Denmark/submit.sh <num_batches> <sims_per_batch> <base_dir>
```

| Parameter | Generic | Denmark |
|-----------|---------|---------|
| Locations | 10 random (pop 5k–50k) | 5 Danish regions (pop 590k–1.86M) |
| R0 | Uniform [2, 8] | Beta(2, 3.5) on [0.5, 4], mode ≈ 1.5 |
| Recovery rate | [0.01, 0.05] /day | [0.07, 0.23] /day (4–14 day infectious period) |
| Sampling rate | [0.00004, 0.00044] /day | [0.001, 0.01] /day |
| Migration rate | [0.00001, 0.0001] | [0.001, 0.012] |
| Sim time | 6–18 (scaled by recovery) | 30–270 days |
| Sample cap | — | 10,000 tips |
| Min tips/location | 10 | 50 |

### Running the Full Pipeline (Single Machine)

```bash
bash stephy2/run_pipeline.sh <input_folder_1> [input_folder_2 ...] <output_folder>
```

### Running Individual Steps

```bash
# Step 1: Analyze tree parameters
python3 stephy2/analyze_trees.py <input_folder>

# Step 2: Build graphs
python3 stephy2/build_graphs.py \
    --input_dir <input_folder> \
    --subtree_width 100 \
    --output <output_folder>/graphs.pt

# Step 3: Train a specific label
python3 stephy2/train.py \
    --graphs <output_folder>/graphs.pt \
    --num_locations 10 \
    --label R0 \
    --output_dir <output_folder>/results_r0
```

### Batch Processing on HPC

```bash
# 1. Inspect data to determine parameters
python3 batch_run/outbreak_check.py <data_dir>

# 2. Build graphs in parallel (one SLURM job per batch)
bash batch_run/submit_build_graphs.sh <data_dir> <subtree_width>

# 3. Merge batch graphs
python3 batch_run/merge_graphs.py --result_dir <output_dir>

# 4. Train all 12 models (3 pipelines x 4 labels)
bash batch_run/submit_train.sh <graphs.pt> <num_locations>

# 5. Evaluate on held-out test set
python3 batch_run/test.py \
    --graphs <test_graphs.pt> \
    --model_dir <train_result_dir> \
    --num_locations <N>
```

## Output Structure

```
output_dir/dataset_name/
├── graphs.pt                    # Shared DGL graph data
├── results_r0/
│   ├── best_model.pt            # Model weights
│   ├── norm_params.pt           # Normalization parameters
│   ├── training_history.csv     # Train/val loss per epoch
│   └── test_predictions.csv     # True vs predicted values
├── results_rr/                  # Recovery Rate results
├── results_sss/                 # Source/Sink Score results
└── results_as/                  # Ancestral State results (classification)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 (graph-level) |
| Max epochs | 500 |
| Early stopping patience | 25 |
| Data split | 80% train / 10% val / 10% test |
| Random seed | 42 |

## Dependencies

```bash
pip install torch dgl numpy pandas scipy numba dendropy scikit-learn tqdm
```

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `dgl` | Deep Graph Library for GNN operations |
| `dendropy` | Phylogenetic tree manipulation |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Scientific computing (KDE, DTW) |
| `numba` | JIT compilation for performance |
| `scikit-learn` | Train/test splitting |
| `tqdm` | Progress bars |

**Additional dependencies for data generation:**

| Package | Purpose |
|---------|---------|
| `polars` | Fast dataframe operations (trajectory parsing) |
| `treeswift` | Tree manipulation (ancestral state extraction) |
| `beast2` | BEAST2 simulation engine (external) |
