# STEPHY trained models

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

## Task naming

The two families use different task directory names, which is historical rather
than meaningful:

| Quantity | `simu/` | `denmark/` |
|---|---|---|
| Basic reproduction number | `reg_r0` | `r0` |
| Recovery rate | `reg_rr` | `rr` |
| Source–sink score | `reg_sss` | `sss` |
| Ancestor location (classification) | `cls_as` | `as` |

`100k_similar_population_result` contains only the source–sink task, because
that is the only quantity that experiment reports.

## Which Denmark head the papers use

Three heads were trained for the Denmark application and compared before one was
selected:

| Internal name | Status |
|---|---|
| `pe_old/stephy` | **Used in both papers.** Point estimates, no conformal prediction. |
| `pe/stephy` | Superseded positional-encoding run. Empirically indistinguishable from `pe_old` at the population level (R₀ correlation 0.98, source–sink 0.97 across all 1,025 clade × source × region cells). Not included in this archive. |
| conformal-prediction variant | Documented out-of-distribution failures on small clades — for example Delta 21I R₀ = 0.055 against 0.90 from `pe_old` on an identical tree. Not included in this archive. |

`pe_old` was kept as the simpler and more stable choice. The four heads under
`denmark/100k_result/pe_old/stephy/` are the exact weights that produced the
Denmark results; byte-identical copies are vendored in the case-study
repository at `models/pe_old_stephy/` so that analysis can be re-run without
this archive.

## What is deliberately not here

`logs/` directories (SLURM scheduler output — provenance noise rather than
results), the superseded `denmark/100k_result/pe` run, the `CBLV-GAT` ablation
under the Denmark application, and `*.prebugfix.bak` backup files, whose
presence would raise avoidable questions about which artefacts produced the
reported numbers.

## Related deposits and repositories

| Resource | Role |
|---|---|
| <https://github.com/leke-lyu/STEPHY> | Method: model, training, simulation engines, figure scripts |
| <https://github.com/leke-lyu/stephy-denmark> | Denmark case study: workflow and results |
| Denmark phylogenetic intermediates (separate Zenodo deposit) | Bootstrap topologies, time-trees, and BEAST2 trees for the case study |

## License

MIT, matching the code repository.
