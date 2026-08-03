# STEPHY trained models

**DOI: [10.5281/zenodo.21766065](https://doi.org/10.5281/zenodo.21766065)**

Trained model weights and test-set predictions for **STEPHY: A Graph Neural
Inference Framework for Rapid Estimation of Regional Epidemic Dynamics from
Large Viral Phylogenies**.

These artefacts are too large for the code repository but are required to
regenerate the paper's figures. Unpack this archive anywhere and point the
`STEPHY_MODELS` environment variable at the resulting directory:

```bash
tar --use-compress-program=unzstd -xf stephy-trained-models.tar.zst
export STEPHY_MODELS="$PWD/stephy-trained-models"
python3 supportingFigures/fig2.py
```

Code: <https://github.com/leke-lyu/STEPHY>

## Contents

```
stephy-trained-models/
├── simu/                                    simulation benchmarks
│   ├── 100k_diverse_population_result/      main benchmark, 100k trees
│   │   ├── stephy/                          STEPHY (graph attention over DTW edges)
│   │   └── CBLV-CNN/                        ablation: CNN only, no graph
│   ├── 100k_similar_population_result/      populations of similar size
│   │   ├── stephy/
│   │   └── CBLV-GAT/                        ablation: graph attention, legacy edges
│   └── 5k_diverse_population_X{1,2,3}_result/   training-set-size replicates
│       ├── stephy/
│       └── CBLV-CNN/
└── denmark/100k_result/pe_old/              Denmark application
    ├── stephy/                              the heads used in the case study
    └── CBLV-CNN/                            ablation
```

Each leaf directory holds one trained task:

| File | Contents |
|---|---|
| `best_model.pt` | Model weights (state dict) |
| `norm_params.pt` | Feature and label normalisation: `aux`, `edge`, `label` |
| `test_predictions.csv` | Held-out predictions used for the reported metrics |
| `training_history.csv` | Per-epoch training and validation loss |
| `cp_calibration.pt`, `cp_metrics.json` | Conformal-prediction calibration, where fitted |

## Related deposits and repositories

The project is published as two code repositories and two data deposits:

| Resource | Role |
|---|---|
| <https://github.com/leke-lyu/STEPHY> | **Code** — the method: model, training, simulation engines, figure scripts |
| <https://github.com/leke-lyu/stephy-denmark> | **Code** — the Denmark case study: subsampling, phylogenetics, and per-clade inference |
| [10.5281/zenodo.21766065](https://doi.org/10.5281/zenodo.21766065) | **Data** — trained models *(this deposit)* |
| [10.5281/zenodo.21766003](https://doi.org/10.5281/zenodo.21766003) | **Data** — Denmark phylogenetic intermediates: bootstrap topologies, time-trees and BEAST2 trees |

## License

MIT, matching the code repository.
