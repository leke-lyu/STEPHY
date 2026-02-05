# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network (GNN)-based framework that translates phylogenetic structure into location-specific epidemiological parameters. It predicts per-location epidemiological metrics (R0, Source/Sink Score, Recovery Rate, Ancestral State) from BEAST2 phylogenetic trees.

## Overview

The repository contains two main pipelines:

| Pipeline | Description |
|----------|-------------|
| **stephy** | Phylogeny-only model using CBLV features extracted from trees |
| **stephy+** | Extended model combining phylogenetic features with epidemiological data |

Both pipelines use a Graph Attention Network (GAT) architecture with edge features derived from Dynamic Time Warping (DTW) of epidemic curves.

## Architecture

### Project Structure

```
STEPHY/
├── stephy/                 # Phylogeny-only pipeline
│   ├── run_pipeline.sh     # Main orchestration script
│   ├── analyze_trees.py    # Tree parameter extraction
│   ├── build_graphs.py     # DGL graph construction
│   ├── train.py            # Training loop
│   ├── model.py            # CBLV_GAT neural network
│   ├── data.py             # Data loading and preprocessing
│   ├── config.py           # Hyperparameters
│   └── beast2_parser.py    # NEXUS file parser
├── stephy+/                # Phylogeny + epidemiological features
│   └── (same structure as stephy/)
├── cblv-cnn/               # Baseline CNN model
│   └── (same structure as stephy/)
└── utils/                  # Shared utilities
    ├── cblv_feature.py
    ├── dtw.py
    ├── edge_feature.py
    ├── graph_loading.py
    ├── node_feature.py
    ├── trajectory_utils.py
    ├── characterizing_outbreak.py
    └── xml_generation.py
```

### Model Architecture (CBLV_GAT)

```
┌─────────────────────────────────────────────────────────────────┐
│                         STEPHY Model                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CBLV (4, W) per node         [stephy+ only: Epi (4) per node] │
│         │                              │                        │
│    CBLVConvEncoder               EpiBranch                      │
│      (3 parallel Conv1d)         (Dense: 4→64→32)               │
│      - Plain branch                    │                        │
│      - Stride branch                   │                        │
│      - Dilate branch                   │                        │
│         │                              │                        │
│    128-dim (stephy)                    │                        │
│     96-dim (stephy+) ──────────────────┘                        │
│                    │                                            │
│              128-dim node embedding                             │
│                    │                                            │
│           GraphEdgeAttention                                    │
│             (Edge features: DTW distance, lag_mean, lag_std)    │
│                    │                                            │
│              256-dim                                            │
│                    │                                            │
│           MLP Classifier: 256 → 128 → 64 → 32 → 1               │
│                    │                                            │
│           Per-node scalar prediction                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: BEAST2 phylogenetic trees + location labels (CSV)
2. **Graph Construction**: One DGL graph per tree
   - Nodes = locations
   - Node features = CBLV encoding (4-channel subtree topology)
   - Edge features = DTW metrics between location epidemic curves
3. **Training**: Separate models for each prediction target
4. **Output**: Per-location predictions with evaluation metrics

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
- Labels: `R0`, `Source_Sink_Score`, `Recovery_Rate`, `Ancestral_State`
- Features (stephy+ only): `Initial_Population`, `Epidemic_Peak`, `Peak_Timing`, `Accumulated_Infections`

## Usage

### Running the Full Pipeline

```bash
# STEPHY (phylogeny-only)
bash stephy/run_pipeline.sh input_folder_1 [input_folder_2 ...] output_folder

# STEPHY+ (phylogeny + epidemiological features)
bash stephy+/run_pipeline.sh input_folder_1 [input_folder_2 ...] output_folder
```

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `analyze_trees.py` | Determine `num_locations` and `subtree_width` |
| 2 | `build_graphs.py` | Build DGL graphs, save to `graphs.pt` |
| 3 | `train.py --label R0` | Train R0 predictor |
| 4 | `train.py --label Source_Sink_Score` | Train source/sink score predictor |
| 5 | `train.py --label Recovery_Rate` | Train recovery rate predictor |
| 6 | `train.py --label Ancestral_State` | Train ancestral state classifier |

### Running Individual Steps

```bash
# Analyze tree parameters
python3 stephy/analyze_trees.py input_folder

# Build graphs
python3 stephy/build_graphs.py \
    --input_dir input_folder \
    --subtree_width 100 \
    --output output_folder/graphs.pt

# Train a specific label
python3 stephy/train.py \
    --graphs output_folder/graphs.pt \
    --num_locations 16 \
    --label R0 \
    --output_dir output_folder/results_r0
```

## Output Structure

```
output_folder/dataset_name/
├── graphs.pt                    # Shared graph data
├── results_r0/
│   ├── best_model.pt            # Model weights
│   ├── norm_params.pt           # Normalization parameters
│   ├── training_history.csv     # Train/val loss per epoch
│   └── test_predictions.csv     # True vs predicted values
├── results_sss/                 # Source/Sink Score results
├── results_rr/                  # Recovery Rate results
└── results_as/                  # Ancestral State results (classification)
```

## Prediction Targets

| Target | Type | Metric |
|--------|------|--------|
| `R0` | Regression | R2, MSE |
| `Source_Sink_Score` | Regression | R2, MSE |
| `Recovery_Rate` | Regression | R2, MSE |
| `Ancestral_State` | Classification | Accuracy |

## Dependencies

Install the required packages:

```bash
pip install torch dgl numpy pandas scipy numba dendropy tqdm
```

### Package List

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `dgl` | Deep Graph Library for GNN operations |
| `dendropy` | Phylogenetic tree manipulation |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Scientific computing (KDE, DTW) |
| `numba` | JIT compilation for performance |
| `tqdm` | Progress bars |

## Key Features

- **CBLV Encoding**: 4-channel compact representation of subtree topology
- **DTW Edge Features**: Captures temporal relationships between location epidemic curves
- **Graph Attention**: Edge-aware attention mechanism for spatial aggregation
- **Multi-task Support**: Separate models for regression and classification targets
- **Normalization**: Per-tree CBLV min-max, training-set Z-score for edges and labels
