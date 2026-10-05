# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network framework that estimates location-specific
epidemiological parameters from a time-scaled phylogeny.

## Repository layout

| Directory | Contents |
|---|---|
| `stephy/` | Primary model (graph attention over DTW edge features) and shared utilities |
| `CBLV-CNN/` | Ablation: CNN only, locations predicted independently |
| `simulate_and_extract/` | Simulation engines and extraction utilities |
| `simulate_and_extract_Denmark/` | Simulation engine for 5 Danish regions |

## Installation

```bash
pip install torch dgl dendropy numpy pandas scipy numba scikit-learn tqdm polars treeswift
```

Simulation also requires BEAST2 with the ReMaster package; see
`simulate_and_extract/simulation_engine.pdf`.

## Quick start

```bash
# simulate: 12 locations, or 5 Danish regions
bash simulate_and_extract/simulate_and_extract_diverse_population.sh <count> [out_dir]
bash simulate_and_extract_Denmark/simulate_and_extract.sh <count> [out_dir]

# train one model per label (flags before positional arguments)
bash stephy/run_pipeline.sh <input_folder> [<input_folder> ...] <output_folder>
bash stephy/run_pipeline.sh --pipeline CBLV-CNN --labels reg_r0,cls_as <input> <output>
```

`run_pipeline.sh` runs `analyze_trees.py` → `build_graphs.py` → `train.py`,
each of which can also be called directly. For SLURM, use
`simulate_and_extract/submit.sh <engine_script> <num_batches> <sims_per_batch> <base_dir>`.

## Simulation engines

| | diverse population | Denmark |
|---|---|---|
| Locations | 12, populations 5k–50k | 5, fixed populations 590k–1.86M |
| R₀ | Uniform [2, 8], ≤ 2 spread within an outbreak | Beta(2, 3.5) on [0.5, 4], ≤ 1 spread |
| Recovery rate γ | [0.05, 0.25] | [0.07, 0.23] |
| Sampling rate δ | [0.0002, 0.0028] | [0.001, 0.01] |
| Migration rate | [0.0005, 0.0045] | [0.001, 0.012] |
| Duration | 6–18 recovery periods | 20–240 time units |
| Stops at | 5,000 samples | 8,000 samples |
| Min tips/location | > 30 | > 30 |

`simulate_and_extract/` also holds variants of the diverse-population script:
`_similar_population` and `_narrow_horizon` (alternative training sets), and
the stress tests `_shift_X{1,2,3}` (population scale), `_heterogeneous_sampling_{a,b}_h`
(δ varying by location or over time) and `_index_loc_{2,3,4}` (multiple
introductions).

## Input data

```
{id}_beast2.trees   # BEAST2 NEXUS; tip 42[&type="I{3}",samp="sample",time=1.5] = location 3, sampled at 1.5
{id}_nf.csv         # one row per location; labels in R0, Recovery_Rate, Source_Sink_Score, Ancestral_State
```

## Prediction targets

One single-task model per label; `reg_` trains with MSE (pinball loss under
conformal prediction), `cls_` with cross-entropy. Without `--labels` all six
are trained.

| Label | Target |
|---|---|
| `reg_r0` | per-location R₀ |
| `reg_rr` | per-location recovery rate γ |
| `reg_sss` | per-location source–sink score |
| `cls_as` | ancestral state: location of the MRCA of all sampled tips |
| `cls_r0` | location with the highest R₀ (experimental) |
| `cls_sss` | location with the highest source–sink score (experimental) |

The three `reg_` models share one architecture and differ only in the
continuous target; likewise the three `cls_` models. The preprint reports
four tasks: `reg_r0`, `reg_rr`, `reg_sss` and `cls_as`. `cls_r0` and `cls_sss`
are experimental options that were not used in any reported analysis.

## Method

**Features.** Each node carries a CBLV matrix (4 × `subtree_width`) encoding
its location's subtree, scaled by tree height, and five log z-scored tree
statistics. Edges carry three DTW features (distance, lag mean, lag std)
between KDE-smoothed tip-time curves.

**Model.** A CNN over the CBLV (96 dims) and an MLP over the statistics
(32 dims) form a 128-dim node embedding. One attention layer, weighted by the
edge features alone, concatenates each node with its neighbour sum; a
256→128→64→32 MLP reads out the prediction.

**Training.** Adam (lr 0.001), batch 32, up to 500 epochs, patience 25 on
`val_loss` (regression) or `val_accuracy` (classification), seed 42. `--seed`
reseeds weight initialisation and batch order for replicate runs; the
train/val/cal/test partition is fixed separately by `split_seed`, so replicates
are scored on one held-out test set.
Conformal prediction is on by default: CQR for regression, RAPS for
classification, α = 0.05, split 80/6.67/6.67/6.67 (80/10/10 without it).

## Output

```
<output_folder>/<input_folder_name>/
├── graphs.pt
└── results_<label>/
    ├── best_model.pt, norm_params.pt, training_history.csv
    ├── test_predictions.csv
    └── cp_calibration.pt, cp_metrics.json      (conformal only)
```

`test_predictions.csv` is keyed by `batch, sim_id, tree_idx` (plus
`location_idx` for regression), so rows join back to `{id}_nf.csv`.

## Data and preprint

This repository holds the code for simulation and training. The data, trained
models and figure scripts are archived on Zenodo:
[10.5281/zenodo.23149954](https://doi.org/10.5281/zenodo.23149954).

Together they support the preprint *STEPHY: A Graph Neural Inference Framework
for Rapid Estimation of Regional Epidemic Dynamics from Large Viral
Phylogenies.* Research Square, 2026. In review.
[10.21203/rs.3.rs-10631496/v2](https://doi.org/10.21203/rs.3.rs-10631496/v2)

## License

MIT — see [LICENSE](LICENSE).
