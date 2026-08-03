# STEPHY

**Spatial Transmission Estimation from PHYlogenies**

STEPHY is a graph neural network that estimates location-specific
epidemiological parameters from a time-scaled phylogeny. Each location becomes
a node encoding its own subtree; the model predicts that location's parameters,
with conformal prediction intervals or sets attached.

This repository supports the research paper *STEPHY: A Graph Neural Inference
Framework for Rapid Estimation of Regional Epidemic Dynamics from Large Viral
Phylogenies*.

## Repository layout

| Directory | Contents |
|---|---|
| `stephy/` | Primary model — graph attention over DTW-derived edge features. Also hosts the shared utilities. |
| `CBLV-CNN/` | Ablation — CNN only, no graph structure; locations predicted independently. |
| `simulate_and_extract/` | Simulation engines (12 locations) and shared extraction utilities. |
| `simulate_and_extract_Denmark/` | Simulation engine for 5 Danish regions. |
| `supportingFigures/` | Scripts that render the paper figures, plus the diagnostics and inference benchmarks behind them. Reads trained models and case-study data from outside the repository — see [Related resources](#related-resources). |
| `zenodo/` | Packaging script and deposit metadata for the trained-model archive. |

## Installation

```bash
pip install torch dgl dendropy numpy pandas scipy numba scikit-learn tqdm polars treeswift
```

Simulation additionally requires BEAST2 with the ReMaster package, installed
separately; see `simulate_and_extract/simulation_engine.pdf`.

## Quick start

```bash
# simulate — 12 locations, or 5 Danish regions
bash simulate_and_extract/simulate_and_extract_diverse_population.sh <count> [out_dir]
bash simulate_and_extract_Denmark/simulate_and_extract.sh <count> [out_dir]

# train one model per label
bash stephy/run_pipeline.sh <input_folder> [<input_folder> ...] <output_folder>
bash stephy/run_pipeline.sh --pipeline CBLV-CNN --labels reg_r0,cls_as <input> <output>
```

Flags must precede the positional arguments. `--pipeline` also accepts
`CBLV-GAT`, a standard `GATConv` ablation that is not included here.

To drive the steps individually:

```bash
python3 stephy/analyze_trees.py <input_folder>       # → num_locations, subtree_width
python3 stephy/build_graphs.py --input_dir <in> --subtree_width 100 --output <out>/graphs.pt
python3 stephy/train.py --graphs <out>/graphs.pt --num_locations 12 \
    --label reg_r0 --output_dir <out>/reg_r0
```

`stephy`'s `graphs.pt` is the superset format, so `CBLV-CNN/train.py` consumes
it directly and ignores the edges.

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

Besides the benchmark engine above, `simulate_and_extract/` carries
`_similar_population.sh` (narrower populations, wider within-outbreak R₀
spread) and `_X1/_X2/_X3.sh`, which scale populations ×1/×2/×3 while scaling
the sampling rate δ inversely to hold tree size roughly constant.

Either engine parallelises over SLURM:

```bash
bash simulate_and_extract/submit.sh <engine_script> <num_batches> <sims_per_batch> <base_dir>
```

## Input data

One pair of files per simulated outbreak:

```
{id}_beast2.trees   # BEAST2 NEXUS, annotated tips
{id}_nf.csv         # one row per location
```

The tip annotation `42[&type="I{3}",samp="sample",time=1.5]` reads as location
3, a sampled tip, sampling time 1.5. Labels come from the `R0`,
`Recovery_Rate`, `Source_Sink_Score` and `Ancestral_State` columns of the CSV,
which carries further columns for downstream analysis. The engines also retain
a `{id}_parameter.csv` of the drawn parameters, which the model does not read.

## Prediction targets

One single-task model per label. The prefix selects the task — `reg_` trains
with MSE, or pinball loss under conformal prediction; `cls_` with
cross-entropy.

| Label | Type | Target |
|---|---|---|
| `reg_r0` | regression | per-location R₀ |
| `reg_rr` | regression | per-location recovery rate γ |
| `reg_sss` | regression | per-location source–sink score |
| `cls_r0` | classification | location with the highest R₀ |
| `cls_sss` | classification | location with the highest source–sink score |
| `cls_as` | classification | index location |

`cls_r0` and `cls_sss` are the argmax of the corresponding regression label.
`cls_as` is the location annotated on the MRCA of all sampled tips, used as the
approximation for the seeding location and reported as the **index location**.
Without `--labels`, `run_pipeline.sh` trains all six.

## Method

**Node features.** A **CBLV** matrix (4 × `subtree_width`) ladderizing the
location's subtree: tip distances from the preceding branch point (channel 0),
branch-point depths (channel 1), and accumulated edge lengths to the parent
branch point, for tips and branch points respectively (channels 2–3). Alongside
it, five **auxiliary statistics** — `mrca_depth`, `earliest_tip_time`,
`latest_tip_time`, `mean_mrca_tip_dist`, `n_tips`.

**Edge features** (`stephy` only). Three **DTW features** from dynamic time
warping between KDE-smoothed tip-time curves — distance, lag mean, lag std.

**Normalization.** CBLV is divided by tree height. Auxiliary features are
log-transformed then z-scored, as are DTW distance and lag std; lag mean gets a
plain z-score. Regression labels are z-scored and predictions
inverse-transformed.

**Model.** A CNN over the CBLV (96 dims) and an MLP over the auxiliary
statistics (32 dims) concatenate into a 128-dim node embedding. A single
attention layer derives its weights from the edge features alone — a 3→16→1 MLP
followed by an edge softmax — and each node is concatenated with its
attention-weighted neighbour sum. A 256→128→64→32→output MLP reads out the
prediction.

**Conformal prediction.** Enabled by default, which changes the split to
80 / 6.67 / 6.67 / 6.67 (train/val/calibration/test). Regression uses **CQR** —
three quantiles trained with pinball loss, calibrated into intervals.
Classification uses **RAPS** — prediction sets built from the softmax with
conformity score `Σ_{k≤r} p_(k) + λ·max(r − k_reg, 0)`. Defaults: `cp_alpha`
0.05, `cqr_quantiles` [0.025, 0.5, 0.975], `raps_lambda` 0.01, `raps_k_reg` 2.

**Training.** Adam at lr 0.001, batch size 32 graphs, up to 500 epochs with
patience 25, seed 42. Early stopping tracks `val_loss` for regression and
`val_accuracy` for classification. Without conformal prediction the split is
80/10/10.

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

## Related resources

This repository holds the method. The trained weights, the empirical
application to Denmark, and the phylogenetic artefacts behind it are published
separately so that each can be cited and versioned on its own.

| Resource | Role |
|---|---|
| [10.5281/zenodo.21766065](https://doi.org/10.5281/zenodo.21766065) | **Trained models.** Weights, normalisation parameters, held-out predictions and training histories for the simulation benchmarks and the Denmark application. Required to regenerate the paper figures. |
| [leke-lyu/stephy-denmark](https://github.com/leke-lyu/stephy-denmark) | **Denmark case study.** Applies STEPHY to SARS-CoV-2 transmission between the five Danish regions: proportional subsampling, the Nextstrain build, bootstrap re-estimation of every tree, and per-clade inference with bootstrap intervals. |
| [10.5281/zenodo.21766003](https://doi.org/10.5281/zenodo.21766003) | **Denmark phylogenetic intermediates.** Bootstrap topologies, time-calibrated trees, ancestral-state reconstructions and the BEAST2 trees fed to STEPHY — roughly three days of compute, archived so the case study reproduces in minutes. |
| [leke-lyu/denmark-ncov](https://github.com/leke-lyu/denmark-ncov) | Browsable Auspice trees for the three Denmark variant builds. |

### Running the figure scripts

`supportingFigures/` resolves those two external roots from the environment
rather than bundling them (see `supportingFigures/_paths.py`). Unset variables
produce a self-describing placeholder path, so a failure names the variable to
set:

```bash
tar --use-compress-program=unzstd -xf stephy-trained-models.tar.zst
export STEPHY_MODELS="$PWD/stephy-trained-models"
export DENMARK_CASE=/path/to/denmark_case
```

| Variable | Needed by |
|---|---|
| `STEPHY_MODELS` | `fig2`, `fig3`, `performance_scatter`, `population_shift_scatter`, `sss_ranking`, `denmark_performance` |
| `DENMARK_CASE` | `fig4`, `fig5`, `denmark_tree_tmrca` |

Two inputs are not redistributable and must be supplied locally: `fig5` needs
GADM level-1 boundaries for Denmark (`gadm41_DNK_1.json`, from
[gadm.org](https://gadm.org/download_country.html), placed beside the script),
and `fig4` needs a local GISAID metadata export. Both scripts exit with
instructions if the file is missing.

## License

MIT — see [LICENSE](LICENSE).
