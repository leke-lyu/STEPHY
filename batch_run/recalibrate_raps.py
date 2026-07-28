#!/usr/bin/env python3
"""
Re-run RAPS calibration for already-trained classification models.

The RAPS conformity score in ``stephy/conformal.py`` previously accumulated the
rank penalty inside the ranking loop, making it grow quadratically in rank
instead of entering once. Coverage stayed valid (calibration and test used the
same score) but the implementation did not match the canonical Angelopoulos
et al. form. The score is now fixed; every ``cp_calibration.pt`` /
``cp_metrics.json`` / ``test_predictions.csv`` produced before that fix still
holds the old numbers and must be regenerated.

Training is NOT repeated. RAPS is post-hoc on a frozen model, so this script
reuses ``best_model.pt`` and ``norm_params.pt`` untouched and only re-runs the
two inference passes (calibration set -> q_hat, test set -> prediction sets).

Reproducing the original split is what makes that sound: ``config.random_seed``
is fixed, ``split_data_cp`` passes it to sklearn as ``random_state``, and
``graph_loader.load_graphs`` iterates ``sorted(glob('batch_*_graphs.pt'))``, so
the same graphs land in the same splits. The script proves this rather than
assuming it -- it aborts unless the regenerated test split matches the
``batch/sim_id/tree_idx`` keys already in ``test_predictions.csv`` and
reproduces ``true_ancestor``/``pred_ancestor``.

Usage:
    # one pipeline, all three classification labels
    python3 recalibrate_raps.py \
        --graphs /projects/lau_projects/simu/100k_diverse_population_result \
        --pipeline stephy --num_locations 12

    # inspect without writing anything
    python3 recalibrate_raps.py --graphs <dir> --pipeline stephy \
        --num_locations 12 --dry_run

Outputs (per label, in <graphs_dir>/<pipeline>/<label>/):
    cp_calibration.pt      rewritten (new q_hat + cal_scores)
    cp_metrics.json        rewritten
    test_predictions.csv   prediction_set / set_size / covered rewritten
Originals are copied to <name>.prebugfix.bak unless --no_backup is given.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score

PIPELINES = ['stephy', 'CBLV-CNN', 'CBLV-GAT']
CLS_LABELS = ['cls_r0', 'cls_sss', 'cls_as']
STEPHY_ROOT = Path(__file__).resolve().parent.parent
BACKUP_SUFFIX = '.prebugfix.bak'


def parse_args():
    p = argparse.ArgumentParser(
        description='Re-run RAPS calibration with the corrected conformity score')
    p.add_argument('--graphs', required=True,
                   help='batch_*_graphs.pt directory (or a single graphs.pt). '
                        'Model dirs are read from <graphs_dir>/<pipeline>/<label>/')
    p.add_argument('--pipeline', required=True, choices=PIPELINES,
                   help='One pipeline per invocation (model/config module names collide)')
    p.add_argument('--labels', default=','.join(CLS_LABELS),
                   help=f'Comma-separated classification labels (default: all of {CLS_LABELS})')
    p.add_argument('--num_locations', type=int, required=True)
    p.add_argument('--model_dir', default=None,
                   help='Override the root holding <pipeline>/<label>/ (default: graphs dir)')
    p.add_argument('--dry_run', action='store_true',
                   help='Compute and report, but write nothing')
    p.add_argument('--no_backup', action='store_true',
                   help='Overwrite in place without keeping *.prebugfix.bak copies')
    p.add_argument('--pred_tolerance', type=float, default=0.001,
                   help='Max fraction of pred_ancestor rows allowed to differ from the '
                        'stored CSV before aborting (default 0.001). Point predictions '
                        'must be reproduced; a tiny drift can come from BLAS '
                        'nondeterminism flipping a near-tied argmax.')
    return p.parse_args()


def setup_imports(pipeline):
    """Put the pipeline's own model/config ahead of the shared stephy modules.

    Mirrors each train.py: `model` and `config` resolve inside the pipeline
    directory, while `model_core`, `graph_loader`, `predictions` and
    `conformal` come from stephy/.
    """
    sys.path.insert(0, str(STEPHY_ROOT / 'stephy'))
    sys.path.insert(0, str(STEPHY_ROOT / pipeline))


def build_model(pipeline, model_config):
    """Instantiate the pipeline's architecture (weights loaded by the caller)."""
    if pipeline == 'CBLV-CNN':
        from model import CBLV_CNN
        return CBLV_CNN(model_config)
    from model import CBLV_GAT
    return CBLV_GAT(model_config)


def make_forward_fn(pipeline, label_name):
    """Return forward_fn(model, batched_g) matching the pipeline's train.py.

    stephy consumes edge features, CBLV-GAT is a plain GAT over the graph, and
    CBLV-CNN ignores the graph structure entirely.
    """
    def forward_fn(model, batched_g):
        node_cblv = batched_g.ndata['cblv']
        node_aux = batched_g.ndata['aux']

        if pipeline == 'stephy':
            predictions = model(batched_g, node_cblv, node_aux, batched_g.edata['feat'])
        elif pipeline == 'CBLV-GAT':
            predictions = model(batched_g, node_cblv, node_aux)
        else:
            predictions = model(node_cblv, node_aux)

        labels = batched_g.ndata[label_name]
        num_nodes_list = batched_g.batch_num_nodes().tolist()
        per_graph_preds = torch.stack(predictions.split(num_nodes_list))
        per_graph_labels = torch.stack(labels.split(num_nodes_list))
        return predictions, labels, per_graph_preds, per_graph_labels

    return forward_fn


def snapshot_raw_features(graph_lists):
    """Record the pre-normalization aux/edge tensors so each label starts clean.

    Normalization rebinds ``g.ndata['aux']`` to a fresh tensor rather than
    mutating in place, so holding the original references is enough to restore
    them -- and lets several labels reuse one expensive graph load without
    compounding normalizations.
    """
    snap = []
    for graphs in graph_lists:
        snap.append([(g.ndata['aux'],
                      g.edata['feat'] if 'feat' in g.edata else None)
                     for g, *_ in graphs])
    return snap


def apply_norm(graph_lists, snapshot, norm):
    """Restore raw features, then apply the saved training normalization.

    Same transforms as train.py: aux is clamp -> log -> z-score; edge channels 0
    (DTW distance) and 2 (lag_std) are log'd before z-scoring, channel 1
    (lag_mean) is z-scored directly.
    """
    aux_mean, aux_std = norm['aux']['mean'], norm['aux']['std']
    edge_norm = norm.get('edge')

    for graphs, snap in zip(graph_lists, snapshot):
        for (g, *_), (raw_aux, raw_edge) in zip(graphs, snap):
            g.ndata['aux'] = (torch.log(raw_aux.clamp(min=1e-8)) - aux_mean) / aux_std

            if edge_norm is not None and raw_edge is not None:
                feat = raw_edge.clone()
                feat[:, 0] = torch.log(feat[:, 0].clamp(min=1e-8))
                feat[:, 2] = torch.log(feat[:, 2].clamp(min=1e-8))
                g.edata['feat'] = (feat - edge_norm['mean']) / edge_norm['std']


def verify_against_stored(csv_path, test_graphs, cp_test, pred_tolerance):
    """Check the regenerated test split reproduces the stored predictions.

    Returns (ok, messages). Metadata keys and true_ancestor must match exactly:
    a mismatch means the graphs or the split moved since training, and nothing
    downstream can be trusted. pred_ancestor is a pure argmax over logits and
    is unaffected by the RAPS fix, so it must also come back -- allowing only
    `pred_tolerance` drift for near-tied argmax under a different BLAS.
    """
    from predictions import build_metadata_df

    msgs = []
    if not csv_path.exists():
        return False, [f'  ABORT  {csv_path.name} not found — cannot verify the split']

    # Read the key columns as text: pandas would otherwise coerce a zero-padded
    # sim_id like "0363" to the integer 363 and fail the comparison against the
    # string the metadata carries.
    stored = pd.read_csv(csv_path, dtype={'batch': str, 'sim_id': str, 'tree_idx': str})
    meta = build_metadata_df(test_graphs, is_classification=True)

    if len(stored) != len(meta):
        return False, [f'  ABORT  row count {len(meta)} != stored {len(stored)}']

    for col in ('batch', 'sim_id', 'tree_idx'):
        lhs = meta[col].astype(str).values
        rhs = stored[col].astype(str).values
        n_bad = int((lhs != rhs).sum())
        if n_bad:
            return False, [f'  ABORT  {n_bad}/{len(lhs)} rows differ in "{col}" — '
                           f'the test split is not the one used at training time']
    msgs.append(f'  split keys   {len(meta)} rows match batch/sim_id/tree_idx exactly')

    n_true_bad = int((cp_test['true_classes'] != stored['true_ancestor'].values).sum())
    if n_true_bad:
        return False, [f'  ABORT  {n_true_bad} rows differ in true_ancestor — '
                       f'labels do not match the stored run']
    msgs.append('  true labels  reproduced exactly')

    n_pred_bad = int((cp_test['pred_classes'] != stored['pred_ancestor'].values).sum())
    frac = n_pred_bad / len(stored)
    if frac > pred_tolerance:
        return False, [f'  ABORT  {n_pred_bad}/{len(stored)} ({frac:.3%}) rows differ in '
                       f'pred_ancestor, above --pred_tolerance {pred_tolerance:.3%}. '
                       f'Point predictions must survive a RAPS-only change.']
    if n_pred_bad:
        msgs.append(f'  point preds  {n_pred_bad}/{len(stored)} ({frac:.3%}) differ — '
                    f'within tolerance, likely near-tied argmax')
    else:
        msgs.append('  point preds  reproduced bit-identically')

    return True, msgs


def backup(path, enabled):
    """Copy a file aside once, preserving the true pre-fix original."""
    if not enabled or not path.exists():
        return
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)


def recalibrate_label(pipeline, label_name, out_dir, config, num_locations,
                      cal_graphs, test_graphs, snapshot, args):
    """Re-run calibration + test application for one label. Returns True if written."""
    from conformal import (calibrate_raps, apply_raps_test, compute_cp_metrics,
                           save_cp_artifacts)
    from predictions import attach_metadata

    print(f'\n{"=" * 72}\n{pipeline} / {label_name}\n{"=" * 72}')

    model_file = out_dir / 'best_model.pt'
    norm_file = out_dir / 'norm_params.pt'
    cal_file = out_dir / 'cp_calibration.pt'
    csv_file = out_dir / 'test_predictions.csv'

    for f in (model_file, norm_file, cal_file):
        if not f.exists():
            print(f'  SKIP   {f.name} not found in {out_dir}')
            return False

    old_cal = torch.load(cal_file, weights_only=False)
    if old_cal.get('method') != 'raps':
        print(f"  SKIP   stored method is '{old_cal.get('method')}', not raps")
        return False

    # Reuse the hyperparameters actually used at training time, not today's
    # config, so the only thing that changes is the score itself.
    alpha = old_cal['alpha']
    lambda_reg = old_cal['lambda_reg']
    k_reg = old_cal['k_reg']
    cfg_alpha = config['train']['cp_alpha']
    if not np.isclose(alpha, cfg_alpha):
        print(f'  NOTE   stored alpha={alpha} differs from config {cfg_alpha}; '
              f'using the stored value')

    apply_norm([cal_graphs, test_graphs], snapshot, torch.load(norm_file, weights_only=False))

    model = build_model(pipeline, config['model'])
    model.load_state_dict(torch.load(model_file, weights_only=False))
    model.eval()

    batch_size = config['train']['batch_size']
    cal_loader = GraphDataLoader([g for g, *_ in cal_graphs], batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader([g for g, *_ in test_graphs], batch_size=batch_size, shuffle=False)
    forward_fn = make_forward_fn(pipeline, label_name)

    print(f'  RAPS   alpha={alpha}, lambda={lambda_reg}, k_reg={k_reg}')
    cp_cal = calibrate_raps(model, cal_loader, label_name, alpha,
                            lambda_reg, k_reg, forward_fn)
    cp_test = apply_raps_test(model, test_loader, label_name, cp_cal['q_hat'],
                              lambda_reg, k_reg, forward_fn)

    ok, msgs = verify_against_stored(csv_file, test_graphs, cp_test, args.pred_tolerance)
    for m in msgs:
        print(m)
    if not ok:
        print('  --> NOT WRITTEN')
        return False

    cp_metrics = compute_cp_metrics(cp_test, is_classification=True,
                                    num_locations=num_locations)
    old_metrics = json.loads((out_dir / 'cp_metrics.json').read_text()) \
        if (out_dir / 'cp_metrics.json').exists() else {}

    def delta(key, fmt='{:.4f}'):
        new = cp_metrics.get(key)
        old = old_metrics.get(key)
        if old is None:
            return fmt.format(new)
        return f'{fmt.format(old)} -> {fmt.format(new)}  ({new - old:+.4f})'

    print(f"  q_hat        {old_cal['q_hat']:.6f} -> {cp_cal['q_hat']:.6f}")
    print(f"  coverage     {delta('empirical_coverage')}   (target {1 - alpha:.2f})")
    print(f"  mean set     {delta('mean_set_size', '{:.3f}')}")
    print(f"  median set   {delta('median_set_size', '{:.1f}')}")
    print(f"  singletons   {delta('singleton_fraction')}")
    print(f"  accuracy     {accuracy_score(cp_test['true_classes'], cp_test['pred_classes']):.4f} "
          f"(unchanged by RAPS)")

    if args.dry_run:
        print('  --> DRY RUN, nothing written')
        return False

    for f in (cal_file, out_dir / 'cp_metrics.json', csv_file):
        backup(f, not args.no_backup)

    save_cp_artifacts(out_dir, cp_cal, cp_metrics, method='raps',
                      alpha=alpha, lambda_reg=lambda_reg, k_reg=k_reg)

    pred_df = pd.DataFrame({
        'true_ancestor': cp_test['true_classes'].astype(int),
        'pred_ancestor': cp_test['pred_classes'].astype(int),
        'prediction_set': [str(s) for s in cp_test['prediction_sets']],
        'set_size': cp_test['set_sizes'].astype(int),
        'covered': cp_test['covered'].astype(int),
    })
    attach_metadata(pred_df, test_graphs, is_classification=True).to_csv(csv_file, index=False)

    print(f'  --> WROTE cp_calibration.pt, cp_metrics.json, test_predictions.csv')
    return True


def main():
    args = parse_args()
    setup_imports(args.pipeline)

    from config import get_config
    from graph_loader import load_graphs
    from conformal import split_data_cp
    from predictions import format_graph_id

    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    bad = [l for l in labels if not l.startswith('cls_')]
    if bad:
        sys.exit(f'Error: RAPS applies to classification only; got {bad}')

    graphs_path = Path(args.graphs)
    model_root = Path(args.model_dir) if args.model_dir else (
        graphs_path if graphs_path.is_dir() else graphs_path.parent)

    config = get_config()
    if not config['train'].get('conformal_prediction', False):
        sys.exit("Error: config has conformal_prediction=False; "
                 "these models were not trained with CP.")

    print(f'Pipeline : {args.pipeline}')
    print(f'Labels   : {", ".join(labels)}')
    print(f'Graphs   : {graphs_path}')
    print(f'Models   : {model_root}/<label>/')
    print(f'Mode     : {"DRY RUN" if args.dry_run else "WRITE"}\n')

    all_graphs = load_graphs(str(graphs_path))
    if args.pipeline == 'CBLV-GAT':
        # CBLV-GAT/train.py adds self-loops immediately after loading, before
        # the split, so the graphs the model saw carry them.
        all_graphs = [(dgl.add_self_loop(g), meta, locs, h)
                      for g, meta, locs, h in all_graphs]

    config['model']['subtree_width'] = all_graphs[0][0].ndata['cblv'].shape[2]
    config['num_locations'] = args.num_locations

    for g, meta, _, _ in all_graphs:
        if g.num_nodes() != args.num_locations:
            sys.exit(f'Error: graph {format_graph_id(meta)} has {g.num_nodes()} '
                     f'locations, expected {args.num_locations}')

    # Same seeding and split call as train.py, so cal/test come back identical.
    np.random.seed(config['train']['random_seed'])
    torch.manual_seed(config['train']['random_seed'])
    train_graphs, val_graphs, cal_graphs, test_graphs = split_data_cp(
        all_graphs, config['train']['train_ratio'], config['train']['random_seed'])
    print(f'Split    : {len(train_graphs)}/{len(val_graphs)}/'
          f'{len(cal_graphs)}/{len(test_graphs)} (train/val/cal/test)')

    # Only cal + test are needed; release the rest before the forward passes.
    del all_graphs, train_graphs, val_graphs
    gc.collect()

    snapshot = snapshot_raw_features([cal_graphs, test_graphs])

    written = []
    for label_name in labels:
        out_dir = model_root / args.pipeline / label_name
        if not out_dir.is_dir():
            print(f'\n{args.pipeline} / {label_name}\n  SKIP   {out_dir} not found')
            continue
        if recalibrate_label(args.pipeline, label_name, out_dir, config,
                             args.num_locations, cal_graphs, test_graphs,
                             snapshot, args):
            written.append(label_name)

    print(f'\n{"=" * 72}')
    if args.dry_run:
        print('Dry run complete — no files modified.')
    else:
        print(f'Rewrote {len(written)}/{len(labels)} label(s): {", ".join(written) or "none"}')
        if written and not args.no_backup:
            print(f'Originals preserved as *{BACKUP_SUFFIX}')
    print('Models, norm_params and training_history were not touched.')


if __name__ == '__main__':
    main()
