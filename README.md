# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network that reads location-specific epidemiological
parameters off a phylogeny. Each location in an outbreak becomes a node; the
model encodes that location's subtree and predicts its epidemiological
parameters, with conformal prediction intervals or sets attached.

## Contents

| Directory | Purpose |
|---|---|
| `stephy/` | Primary model — edge-attention GAT over DTW-derived edge features. Also hosts the utilities shared with the ablation. |
| `CBLV-CNN/` | Ablation — CNN only, no graph structure; each location predicted independently. |
| `simulate_and_extract/` | Simulation engines (12 locations) + shared extraction utilities. |
| `simulate_and_extract_Denmark/` | Simulation engine parameterised for 5 Danish regions. |

`stephy/run_pipeline.sh` also accepts `--pipeline CBLV-GAT` (a standard
`dgl.nn.GATConv` ablation without DTW edge features). That variant is not
included here; the flag will report a missing directory.

## Prediction targets

Six single-task models. The prefix selects the task type — `reg_` trains with
MSE (or pinball loss under conformal prediction), `cls_` with cross-entropy.

| Label | Type | Predicts |
|---|---|---|
| `reg_r0` | regression | per-location R₀ |
| `reg_rr` | regression | per-location recovery rate γ |
| `reg_sss` | regression | per-location source–sink score |
| `cls_r0` | classification | which location has the highest R₀ |
| `cls_sss` | classification | which location is the biggest exporter |
| `cls_as` | classification | ancestral state — which location sits at the root of the sampled phylogeny |

`cls_r0` and `cls_sss` are the argmax of the corresponding regression label.
`cls_as` is the location annotated on the MRCA of all sampled tips, which is
not necessarily the location the outbreak was seeded in — the simulation
records that separately as `Spillover_Loc`.

## Architecture

```
CBLV (4 × W) per node              Aux (5 tree stats) per node
       │                                    │
 CBLVConvEncoder                        AuxBranch
   3 parallel Conv1d                    5 → 64 → 32
   plain  [12,24,48]                        │
   stride [12,24]                           │
   dilate [12,24]                           │
       └────── 96-dim ──────┬────── 32-dim ─┘
                            │
                   128-dim node embedding
                            │
                  GraphEdgeAttention
                    edge features: DTW (distance, lag_mean, lag_std)
                    attention MLP 3 → 16 → 1
                            │
                256-dim (128 self ‖ 128 aggregated)
                            │
                MLP 256 → 128 → 64 → 32 → out
                            │
                   per-location prediction
```

`out` is 1 for a point estimate, or 3 when conformal prediction is enabled for
a regression label (the CQR quantiles). Classification emits one logit per
location node, so a graph's node outputs form the class scores directly.

**`CBLV-CNN`** widens the CNN to 96+48+48 = 192 and the aux branch to
5 → 128 → 64, reaching the same 256-dim classifier input without the graph
layer. It isolates what spatial aggregation contributes.

## Features

**CBLV** — 4 channels per node, ladderized over the location's subtree:

| Channel | Value |
|---|---|
| 0 | tip distance from the preceding branch point (the first entry instead carries the tip's full root distance) |
| 1 | branch-point depth (root distance) |
| 2 | accumulated edge length from each tip to its parent branch point |
| 3 | accumulated edge length from each branch point to its parent |

Padded or truncated to a fixed `subtree_width`.

**Auxiliary tree statistics** — 5 per node: `mrca_depth`, `earliest_tip_time`,
`latest_tip_time`, `mean_mrca_tip_dist`, `n_tips`.

**DTW edge features** — 3 per edge (`stephy` only), from dynamic time warping
between KDE-smoothed tip-time curves: distance, lag mean, lag std. The lag
terms are scaled by the KDE grid spacing.

**Normalization** — CBLV divided by tree height (or `log1p`); aux log then
z-score; edge features log+z-score for distance and lag std, plain z-score for
lag mean; regression labels z-scored, with predictions inverse-transformed.

## Conformal prediction

On by default (`conformal_prediction` in `config.py`). The split becomes
80 / 6.67 / 6.67 / 6.67 (train/val/calibration/test) instead of 80/10/10.

- **Regression — CQR.** The model emits 3 quantiles trained with pinball loss;
  the calibration set fixes an additive correction giving prediction intervals.
- **Classification — RAPS.** Post-hoc on the softmax outputs, giving prediction
  sets. Conformity score `Σ_{k≤r} p_(k) + λ·max(r − k_reg, 0)`, with the rank
  penalty applied once at the true class's rank.

Defaults: `cp_alpha` 0.05 (95% coverage), `cqr_quantiles` [0.025, 0.5, 0.975],
`raps_lambda` 0.01, `raps_k_reg` 2.

## Input data

One pair of files per simulated outbreak:

```
input_folder/
  {id}_beast2.trees   # BEAST2 NEXUS, annotated tips
  {id}_nf.csv         # one row per location
```

Tip annotation — `42[&type="I{3}",samp="sample",time=1.5]` reads as location 3,
a sampled tip, sampling time 1.5.

The columns of `{id}_nf.csv` consumed by the model are `Location`, `R0`,
`Recovery_Rate`, `Source_Sink_Score` (exports − imports over their sum, in
[−1, 1]) and `Ancestral_State` (1 for the MRCA location of the sampled tips).
The simulation also writes `Initial_Population`, `Epidemic_Peak`,
`Peak_Timing`, `Accumulated_Infections` and `Spillover_Loc`, which are carried
for analysis but not used as labels.

## Usage

### Generate data

```bash
# 12 locations, diverse population sizes — the benchmark scenario
bash simulate_and_extract/simulate_and_extract_diverse_population.sh <target_count> [output_dir]

# 5 Danish regions with real population sizes
bash simulate_and_extract_Denmark/simulate_and_extract.sh <target_count> [output_dir]

# either engine, parallelised over SLURM
bash simulate_and_extract/submit.sh <engine_script> <num_batches> <sims_per_batch> <base_dir>
```

`simulate_and_extract/` carries five engines: the benchmark above,
`_similar_population.sh` (narrow population range, wider R₀ spread), and
`_X1/_X2/_X3.sh`, which scale populations ×1/×2/×3 while scaling the sampling
rate δ inversely to hold tree size roughly constant.

| | diverse population | Denmark |
|---|---|---|
| Locations | 12, populations 5k–50k | 5, populations 590k–1.86M |
| R₀ | Uniform [2, 8], max spread 2 | Beta(2, 3.5) on [0.5, 4] |
| Recovery rate γ | [0.05, 0.25] | [0.07, 0.23] |
| Sampling rate δ | [0.0002, 0.0028] | [0.001, 0.01] |
| Migration rate | [0.0005, 0.0045] | [0.001, 0.012] |
| Duration | 6–18 recovery periods | 20–240 time units |
| Stops at | 5,000 samples | 8,000 samples |
| Min tips/location | > 30 | > 30 |

### Train

```bash
# all six labels, primary model
bash stephy/run_pipeline.sh <input_folder> [<input_folder> ...] <output_folder>

# ablation, or a subset of labels
bash stephy/run_pipeline.sh --pipeline CBLV-CNN <input_folders...> <output_folder>
bash stephy/run_pipeline.sh --labels reg_r0,cls_as <input_folders...> <output_folder>
```

`stephy`'s `graphs.pt` is the superset format — fully connected edges with DTW
features plus all node features — so `CBLV-CNN/train.py` can be pointed at it
directly and will simply ignore the edges. (`run_pipeline.sh` rebuilds graphs
on each invocation; reuse it via `train.py` if you want to build only once.)

### Individual steps

```bash
python3 stephy/analyze_trees.py <input_folder>          # → num_locations, subtree_width

python3 stephy/build_graphs.py \
    --input_dir <input_folder> --subtree_width 100 --output <out>/graphs.pt

python3 stephy/train.py \
    --graphs <out>/graphs.pt --num_locations 12 \
    --label reg_r0 --output_dir <out>/reg_r0
```

## Output

`run_pipeline.sh` creates one directory per input folder, named after it:

```
<output_folder>/<input_folder_name>/
├── graphs.pt
└── results_<label>/                  # one per trained label
    ├── best_model.pt                 # weights at the best epoch
    ├── norm_params.pt                # aux and label normalization statistics (plus edge, for stephy)
    ├── training_history.csv          # per-epoch train/validation loss, plus val_accuracy for cls_ labels
    ├── test_predictions.csv          # per-test-graph predictions
    ├── cp_calibration.pt             # q_hat + calibration scores   (conformal only)
    └── cp_metrics.json               # coverage, interval width or set size (conformal only)
```

`test_predictions.csv` is keyed by `batch, sim_id, tree_idx` (plus
`location_idx, location_name` for regression), so rows join back to the source
`{id}_nf.csv` and across labels.

## Training configuration

| | |
|---|---|
| Optimizer | Adam, lr 0.001 |
| Batch size | 32 graphs |
| Max epochs | 500, early stopping patience 25 |
| Early stopping on | `val_loss` (regression), `val_accuracy` (classification) |
| Split | 80/10/10, or 80/6.67/6.67/6.67 with conformal prediction |
| Random seed | 42 |

## Dependencies

```bash
pip install torch dgl dendropy numpy pandas scipy numba scikit-learn tqdm polars treeswift
```

| Package | Used for |
|---|---|
| `torch`, `dgl` | model and graph operations |
| `dendropy`, `treeswift` | phylogenetic tree manipulation |
| `numpy`, `pandas`, `polars` | numerics and trajectory parsing |
| `scipy`, `numba` | KDE and DTW, with JIT compilation |
| `scikit-learn` | data splitting and metrics |
| `tqdm` | progress reporting |

BEAST2 with the ReMaster package is required for simulation and must be
installed separately; see `simulate_and_extract/simulation_engine.pdf`.
