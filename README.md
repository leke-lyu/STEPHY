# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network that reads location-specific epidemiological
parameters off a phylogeny. Each location becomes a node encoding its own
subtree; the model predicts that location's parameters, with conformal
prediction intervals or sets attached.

This repository supports the research paper *STEPHY: A Graph Neural Inference
Framework for Rapid Estimation of Regional Epidemic Dynamics from Large Viral
Phylogenies*. The manuscript has not yet been submitted to a journal or posted
as a preprint, so no link is available; this section will carry one once it is.

## Contents

| Directory | Purpose |
|---|---|
| `stephy/` | Primary model — edge-attention GAT over DTW-derived edge features. Also hosts the shared utilities. |
| `CBLV-CNN/` | Ablation — CNN only, no graph structure; locations predicted independently. |
| `simulate_and_extract/` | Simulation engines (12 locations) + shared extraction utilities. |
| `simulate_and_extract_Denmark/` | Simulation engine for 5 Danish regions. |
| `supportingFigures/` | Scripts that render the figures in the paper draft, plus the diagnostics and inference benchmarks behind them. They read the trained-model output described below. |

`run_pipeline.sh` also accepts `--pipeline CBLV-GAT`, a standard `GATConv`
ablation that is not included here.

Each script in `supportingFigures/` takes its input paths as `argparse`
arguments whose defaults point at the author's local directories, so pass
`--base_dir` (and the script's other path flags) to run them elsewhere. Only
the scripts are versioned; the PDFs they render are not.

## Prediction targets

One single-task model per label. The prefix selects the task — `reg_` trains
with MSE (pinball loss under conformal prediction), `cls_` with cross-entropy.

| Label | Type | Predicts |
|---|---|---|
| `reg_r0` | regression | per-location R₀ |
| `reg_rr` | regression | per-location recovery rate γ |
| `reg_sss` | regression | per-location source–sink score |
| `cls_as` | classification | index location — the location at the root of the sampled phylogeny |

`cls_as` is the location annotated on the MRCA of all sampled tips, used as the
approximation for the seeding location and reported as the **index location**.
Two further labels, `cls_r0` and `cls_sss`
(argmax of the corresponding regression label), are also accepted; without
`--labels`, `run_pipeline.sh` trains all six.

## Features

Per node: a **CBLV** matrix (4 × `subtree_width`) ladderizing the location's
subtree — tip and branch-point depths in channels 0–1, accumulated edge
lengths to the parent branch point in channels 2–3 — plus five **auxiliary
statistics** (`mrca_depth`, `earliest_tip_time`, `latest_tip_time`,
`mean_mrca_tip_dist`, `n_tips`).

Per edge (`stephy` only): three **DTW features** from dynamic time warping
between KDE-smoothed tip-time curves — distance, lag mean, lag std.

CBLV is divided by tree height; aux and edge features are log-transformed then
z-scored (plain z-score for lag mean); regression labels are z-scored and
predictions inverse-transformed.

## Conformal prediction

Enabled by default, which changes the split to 80 / 6.67 / 6.67 / 6.67
(train/val/calibration/test). Regression uses **CQR** — three quantiles trained
with pinball loss, calibrated into intervals. Classification uses **RAPS** —
prediction sets built from the softmax with conformity score
`Σ_{k≤r} p_(k) + λ·max(r − k_reg, 0)`. Defaults: `cp_alpha` 0.05,
`cqr_quantiles` [0.025, 0.5, 0.975], `raps_lambda` 0.01, `raps_k_reg` 2.

## Input data

One pair of files per simulated outbreak:

```
{id}_beast2.trees   # BEAST2 NEXUS, annotated tips
{id}_nf.csv         # one row per location
```

The tip annotation `42[&type="I{3}",samp="sample",time=1.5]` reads as
location 3, a sampled tip, sampling time 1.5. Labels come from the `R0`,
`Recovery_Rate`, `Source_Sink_Score` and `Ancestral_State` columns; the file
carries further columns for downstream analysis.

## Usage

```bash
# generate data — 12 locations, or 5 Danish regions
bash simulate_and_extract/simulate_and_extract_diverse_population.sh <count> [out_dir]
bash simulate_and_extract_Denmark/simulate_and_extract.sh <count> [out_dir]

# parallelise either engine over SLURM
bash simulate_and_extract/submit.sh <engine_script> <num_batches> <sims_per_batch> <base_dir>

# train
bash stephy/run_pipeline.sh <input_folder> [<input_folder> ...] <output_folder>
bash stephy/run_pipeline.sh --pipeline CBLV-CNN --labels reg_r0,cls_as <input> <output>
```

Individual steps:

```bash
python3 stephy/analyze_trees.py <input_folder>       # → num_locations, subtree_width
python3 stephy/build_graphs.py --input_dir <in> --subtree_width 100 --output <out>/graphs.pt
python3 stephy/train.py --graphs <out>/graphs.pt --num_locations 12 \
    --label reg_r0 --output_dir <out>/reg_r0
```

`stephy`'s `graphs.pt` is the superset format, so `CBLV-CNN/train.py` can
consume it directly and ignore the edges.

Besides the benchmark engine above, `simulate_and_extract/` carries
`_similar_population.sh` (narrow population range, wider R₀ spread) and
`_X1/_X2/_X3.sh`, which scale populations ×1/×2/×3 while scaling the sampling
rate δ inversely to hold tree size roughly constant.

| | diverse population | Denmark |
|---|---|---|
| Locations | 12, populations 5k–50k | 5, populations 590k–1.86M |
| R₀ | Uniform [2, 8] | Beta(2, 3.5) on [0.5, 4] |
| Recovery rate γ | [0.05, 0.25] | [0.07, 0.23] |
| Sampling rate δ | [0.0002, 0.0028] | [0.001, 0.01] |
| Migration rate | [0.0005, 0.0045] | [0.001, 0.012] |
| Duration | 6–18 recovery periods | 20–240 time units |
| Stops at | 5,000 samples | 8,000 samples |
| Min tips/location | > 30 | > 30 |

## Output

`run_pipeline.sh` writes one directory per input folder:

```
<output_folder>/<input_folder_name>/
├── graphs.pt
└── results_<label>/
    ├── best_model.pt, norm_params.pt, training_history.csv
    ├── test_predictions.csv
    └── cp_calibration.pt, cp_metrics.json      (conformal only)
```

`test_predictions.csv` is keyed by `batch, sim_id, tree_idx` (plus
`location_idx, location_name` for regression), so rows join back to
`{id}_nf.csv` and across labels.

## Configuration

Adam at lr 0.001, batch size 32 graphs, up to 500 epochs with patience 25,
seed 42. Early stopping tracks `val_loss` for regression and `val_accuracy`
for classification. Without conformal prediction the split is 80/10/10.

## Dependencies

```bash
pip install torch dgl dendropy numpy pandas scipy numba scikit-learn tqdm polars treeswift
```

BEAST2 with the ReMaster package must be installed separately for simulation;
see `simulate_and_extract/simulation_engine.pdf`.
