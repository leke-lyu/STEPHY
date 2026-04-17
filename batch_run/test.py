#!/usr/bin/env python3
"""
Test trained models on a new dataset.

Evaluates all 18 pipeline x label combinations (3 pipelines x 6 labels)
using pre-trained models from a result directory.  For each combination,
loads the saved model and normalization parameters, applies them to the
test graphs, and writes per-combination predictions plus an overall summary.

Supports conformal prediction: if cp_calibration.pt exists alongside a model,
CQR intervals (regression) or RAPS prediction sets (classification) are
applied to test predictions.

Usage:
    python3 test.py --graphs <graphs.pt> --model_dir <result_dir> --num_locations <N>

Expected layout under model_dir:
    <model_dir>/<pipeline>/<label>/best_model.pt
    <model_dir>/<pipeline>/<label>/norm_params.pt
    <model_dir>/<pipeline>/<label>/cp_calibration.pt  (optional, if CP was used)
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


PIPELINES = ['stephy', 'CBLV-CNN', 'CBLV-GAT']
LABELS = ['reg_r0', 'cls_r0', 'reg_rr', 'reg_sss', 'cls_sss', 'cls_as']
STEPHY_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(STEPHY_ROOT / 'stephy'))
from graph_loader import load_graphs
from predictions import attach_metadata, format_graph_id
from conformal import load_cp_calibration, compute_cp_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Test trained models on new dataset')
    parser.add_argument('--graphs', required=True, help='Path to new dataset graphs.pt')
    parser.add_argument('--model_dir', required=True, help='Path to directory containing trained models')
    parser.add_argument('--num_locations', type=int, required=True, help='Number of locations')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_module(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model_and_config(pipeline):
    """Return (model_class, config_dict) for a pipeline."""
    pipeline_dir = STEPHY_ROOT / pipeline
    model_mod = _load_module(f'{pipeline}_model', pipeline_dir / 'model.py')
    config_mod = _load_module(f'{pipeline}_config', pipeline_dir / 'config.py')
    config = config_mod.get_config()

    if pipeline == 'CBLV-CNN':
        return model_mod.CBLV_CNN, config
    else:  # stephy and CBLV-GAT both export CBLV_GAT
        return model_mod.CBLV_GAT, config


# ---------------------------------------------------------------------------
# Normalization (mirrors each pipeline's train.py)
# ---------------------------------------------------------------------------

def normalize_aux(graphs, params):
    """Log-transform and z-score standardize auxiliary node features in-place."""
    mean, std = params['mean'], params['std']
    for g, *_ in graphs:
        g.ndata['aux'] = (torch.log(g.ndata['aux'].clamp(min=1e-8)) - mean) / std


def normalize_edge(graphs, params):
    """Log-transform columns 0 and 2, then standardize edge features in-place."""
    mean, std = params['mean'], params['std']
    for g, *_ in graphs:
        feat = g.edata['feat'].clone()
        feat[:, 0] = torch.log(feat[:, 0].clamp(min=1e-8))
        feat[:, 2] = torch.log(feat[:, 2].clamp(min=1e-8))
        g.edata['feat'] = (feat - mean) / std


def normalize_labels(graphs, label_name, params):
    """Standardize node-level labels in-place."""
    mean, std = params['mean'], params['std']
    for g, *_ in graphs:
        g.ndata[label_name] = (g.ndata[label_name] - mean) / std


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward(model, batched_g, label_name, pipeline):
    """Run a forward pass and return flat and per-graph predictions/labels."""
    node_cblv = batched_g.ndata['cblv']
    node_aux = batched_g.ndata['aux']

    if pipeline == 'stephy':
        preds = model(batched_g, node_cblv, node_aux, batched_g.edata['feat'])
    elif pipeline == 'CBLV-CNN':
        preds = model(node_cblv, node_aux)
    else:  # CBLV-GAT
        preds = model(batched_g, node_cblv, node_aux)

    labels = batched_g.ndata[label_name]
    splits = batched_g.batch_num_nodes().tolist()
    return preds, labels, torch.stack(preds.split(splits)), torch.stack(labels.split(splits))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, label_name, is_classification, pipeline):
    """Run inference over the full dataloader."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batched_g in dataloader:
            flat_preds, flat_labels, pg_preds, pg_labels = _forward(
                model, batched_g, label_name, pipeline)

            if is_classification:
                all_preds.extend(pg_preds.argmax(dim=1).numpy())
                all_labels.extend(pg_labels.argmax(dim=1).numpy())
            else:
                all_preds.extend(flat_preds.numpy())
                all_labels.extend(flat_labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# CQR test-time application
# ---------------------------------------------------------------------------

def apply_cqr(model, dataloader, label_name, pipeline, q_hat, label_norm):
    """Apply CQR to produce conformalized intervals. Returns dict in true scale."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batched_g in dataloader:
            flat_preds, flat_labels, _, _ = _forward(
                model, batched_g, label_name, pipeline)
            all_preds.append(flat_preds.numpy())
            all_labels.append(flat_labels.numpy())

    preds = np.concatenate(all_preds)    # (N, 3)
    labels = np.concatenate(all_labels)  # (N,)

    q_lo = preds[:, 0]
    q_mid = preds[:, 1]
    q_hi = preds[:, 2]

    # Handle quantile crossing
    q_lo, q_hi = np.minimum(q_lo, q_hi), np.maximum(q_lo, q_hi)

    lower_z = q_lo - q_hat
    upper_z = q_hi + q_hat

    if label_norm is not None:
        lm = label_norm['mean'].item()
        ls = label_norm['std'].item()
        lower = lower_z * ls + lm
        upper = upper_z * ls + lm
        point = q_mid * ls + lm
        true = labels * ls + lm
    else:
        lower, upper, point, true = lower_z, upper_z, q_mid, labels

    width = upper - lower
    covered = ((true >= lower) & (true <= upper)).astype(float)

    return {
        'true': true, 'point': point,
        'lower': lower, 'upper': upper,
        'width': width, 'covered': covered,
    }


# ---------------------------------------------------------------------------
# RAPS test-time application
# ---------------------------------------------------------------------------

def apply_raps(model, dataloader, label_name, pipeline, q_hat, lambda_reg, k_reg):
    """Apply RAPS to produce prediction sets. Returns dict."""
    model.eval()
    all_true, all_pred, all_sets, all_sizes, all_covered = [], [], [], [], []

    with torch.no_grad():
        for batched_g in dataloader:
            _, _, pg_preds, pg_labels = _forward(
                model, batched_g, label_name, pipeline)
            probs = F.softmax(pg_preds, dim=1)
            true_classes = pg_labels.argmax(dim=1)
            pred_classes = pg_preds.argmax(dim=1)

            for i in range(probs.shape[0]):
                p = probs[i]
                true_cls = true_classes[i].item()
                sorted_probs, sorted_idx = torch.sort(p, descending=True)

                pred_set = []
                cumsum = 0.0
                for j in range(len(sorted_probs)):
                    cumsum += sorted_probs[j].item()
                    cumsum += lambda_reg * max(j + 1 - k_reg, 0)
                    pred_set.append(sorted_idx[j].item())
                    if cumsum >= q_hat:
                        break

                all_true.append(true_cls)
                all_pred.append(pred_classes[i].item())
                all_sets.append(sorted(pred_set))
                all_sizes.append(len(pred_set))
                all_covered.append(1.0 if true_cls in pred_set else 0.0)

    return {
        'true_classes': np.array(all_true),
        'pred_classes': np.array(all_pred),
        'prediction_sets': all_sets,
        'set_sizes': np.array(all_sizes),
        'covered': np.array(all_covered),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, label_name, is_classification,
                    label_norm, num_locations):
    """Compute evaluation metrics and build a predictions DataFrame."""
    if is_classification:
        acc = accuracy_score(labels, preds)
        metrics = {'accuracy': acc}
        for c in range(num_locations):
            mask = labels == c
            if mask.sum() > 0:
                metrics[f'class_{c}_accuracy'] = float((preds[mask] == c).mean())

        pred_df = pd.DataFrame({
            'true_ancestor': labels.astype(int),
            'pred_ancestor': preds.astype(int),
        })
        return metrics, pred_df

    # Regression: inverse-transform to original scale
    if label_norm is not None:
        lm = label_norm['mean'].item()
        ls = label_norm['std'].item()
        preds = preds * ls + lm
        labels = labels * ls + lm

    r2 = r2_score(labels, preds)
    mse = mean_squared_error(labels, preds)
    metrics = {'r2': r2, 'mse': mse}

    pred_df = pd.DataFrame({
        f'true_{label_name}': labels,
        f'pred_{label_name}': preds,
    })
    return metrics, pred_df


# ---------------------------------------------------------------------------
# Per-combination test
# ---------------------------------------------------------------------------

def test_one(pipeline, label_name, raw_graphs, model_dir, num_locations, output_dir):
    """Test one pipeline x label combination. Returns metrics dict or None."""
    model_path = model_dir / pipeline / label_name
    is_classification = label_name.startswith('cls_')

    best_model_file = model_path / 'best_model.pt'
    norm_file = model_path / 'norm_params.pt'
    if not best_model_file.exists():
        print(f"    SKIP: {best_model_file} not found")
        return None
    if not norm_file.exists():
        print(f"    SKIP: {norm_file} not found")
        return None

    # Check for CP calibration
    cp_cal = load_cp_calibration(model_path)
    has_cp = cp_cal is not None

    # Deep copy so in-place normalizations don't leak across combinations.
    graphs = deepcopy(raw_graphs)

    # CBLV-GAT: add self-loops (matches training)
    if pipeline == 'CBLV-GAT':
        graphs = [(dgl.add_self_loop(g), *rest) for g, *rest in graphs]

    # Apply training-set normalization
    norm_params = torch.load(norm_file, weights_only=False)
    normalize_aux(graphs, norm_params['aux'])
    if pipeline == 'stephy' and 'edge' in norm_params:
        normalize_edge(graphs, norm_params['edge'])

    label_norm = norm_params.get('label')
    if not is_classification and label_norm is not None:
        normalize_labels(graphs, label_name, label_norm)

    dataloader = GraphDataLoader([g for g, *_ in graphs], batch_size=32, shuffle=False)

    model_cls, config = get_model_and_config(pipeline)
    config['model']['subtree_width'] = graphs[0][0].ndata['cblv'].shape[2]

    # CQR models have 3 outputs
    if has_cp and cp_cal['method'] == 'cqr':
        config['model']['num_outputs'] = 3

    model = model_cls(config['model'])
    model.load_state_dict(torch.load(best_model_file, weights_only=False))

    # --- Standard evaluation ---
    preds, labels = evaluate(model, dataloader, label_name, is_classification, pipeline)
    metrics, pred_df = compute_metrics(preds, labels, label_name, is_classification,
                                       label_norm, num_locations)

    # --- Conformal prediction ---
    if has_cp:
        q_hat = cp_cal['q_hat']

        if cp_cal['method'] == 'cqr':
            cp_test = apply_cqr(model, dataloader, label_name, pipeline, q_hat, label_norm)
            cp_metrics = compute_cp_metrics(cp_test, is_classification=False)
            metrics['cp_coverage'] = cp_metrics['empirical_coverage']
            metrics['cp_mean_width'] = cp_metrics['mean_interval_width']

            pred_df = pd.DataFrame({
                f'true_{label_name}': cp_test['true'],
                f'pred_{label_name}': cp_test['point'],
                f'lower_{label_name}': cp_test['lower'],
                f'upper_{label_name}': cp_test['upper'],
                'interval_width': cp_test['width'],
                'covered': cp_test['covered'].astype(int),
            })

        elif cp_cal['method'] == 'raps':
            lambda_reg = cp_cal.get('lambda_reg', 0.01)
            k_reg = cp_cal.get('k_reg', 2)
            cp_test = apply_raps(model, dataloader, label_name, pipeline,
                                 q_hat, lambda_reg, k_reg)
            cp_metrics = compute_cp_metrics(cp_test, is_classification=True,
                                            num_locations=num_locations)
            metrics['cp_coverage'] = cp_metrics['empirical_coverage']
            metrics['cp_mean_set_size'] = cp_metrics['mean_set_size']

            pred_df = pd.DataFrame({
                'true_ancestor': cp_test['true_classes'].astype(int),
                'pred_ancestor': cp_test['pred_classes'].astype(int),
                'prediction_set': [str(s) for s in cp_test['prediction_sets']],
                'set_size': cp_test['set_sizes'].astype(int),
                'covered': cp_test['covered'].astype(int),
            })

        # Save CP metrics for this test run
        cp_metrics_file = output_dir / pipeline / label_name / 'cp_metrics.json'
        cp_metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_metrics_file, 'w') as f:
            json.dump(cp_metrics, f, indent=2)

    save_dir = output_dir / pipeline / label_name
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_df = attach_metadata(pred_df, graphs, is_classification)
    pred_df.to_csv(save_dir / 'test_predictions.csv', index=False)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    graphs_path = Path(args.graphs)
    model_dir = Path(args.model_dir)
    output_dir = graphs_path if graphs_path.is_dir() else graphs_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graphs from {graphs_path}...")
    raw_graphs = load_graphs(args.graphs)
    print(f"Loaded {len(raw_graphs)} graphs, {args.num_locations} locations")

    for g, meta, locs, _ in raw_graphs:
        if g.num_nodes() != args.num_locations:
            raise ValueError(
                f"Graph {format_graph_id(meta)}: expected {args.num_locations} locations, "
                f"got {g.num_nodes()}."
            )

    summary_rows = []

    for pipeline in PIPELINES:
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline}")
        print(f"{'='*60}")

        for label_name in LABELS:
            is_cls = label_name.startswith('cls_')
            print(f"\n  {label_name} ({'classification' if is_cls else 'regression'})...")

            metrics = test_one(pipeline, label_name, raw_graphs, model_dir,
                               args.num_locations, output_dir)
            if metrics is None:
                continue

            if is_cls:
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                for c in range(args.num_locations):
                    key = f'class_{c}_accuracy'
                    if key in metrics:
                        print(f"      Class {c}: {metrics[key]:.4f}")
                if 'cp_coverage' in metrics:
                    print(f"    CP Coverage: {metrics['cp_coverage']:.4f}")
                    print(f"    CP Mean Set Size: {metrics['cp_mean_set_size']:.2f}")
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'accuracy', 'metric_value': metrics['accuracy'],
                })
                if 'cp_coverage' in metrics:
                    summary_rows.append({
                        'pipeline': pipeline, 'label': label_name,
                        'metric_name': 'cp_coverage', 'metric_value': metrics['cp_coverage'],
                    })
                    summary_rows.append({
                        'pipeline': pipeline, 'label': label_name,
                        'metric_name': 'cp_mean_set_size', 'metric_value': metrics['cp_mean_set_size'],
                    })
            else:
                print(f"    R2: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")
                if 'cp_coverage' in metrics:
                    print(f"    CP Coverage: {metrics['cp_coverage']:.4f}")
                    print(f"    CP Mean Width: {metrics['cp_mean_width']:.4f}")
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'r2', 'metric_value': metrics['r2'],
                })
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'mse', 'metric_value': metrics['mse'],
                })
                if 'cp_coverage' in metrics:
                    summary_rows.append({
                        'pipeline': pipeline, 'label': label_name,
                        'metric_name': 'cp_coverage', 'metric_value': metrics['cp_coverage'],
                    })
                    summary_rows.append({
                        'pipeline': pipeline, 'label': label_name,
                        'metric_name': 'cp_mean_width', 'metric_value': metrics['cp_mean_width'],
                    })

    # Summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / 'summary.csv', index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {output_dir / 'summary.csv'}")
        print(summary_df.to_string(index=False))

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
