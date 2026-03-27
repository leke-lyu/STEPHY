#!/usr/bin/env python3
"""
Test trained models on a new dataset.

Evaluates all 12 pipeline x label combinations (3 pipelines x 4 labels)
using pre-trained models from a result directory.  For each combination,
loads the saved model and normalization parameters, applies them to the
test graphs, and writes per-combination predictions plus an overall summary.

Usage:
    python3 test.py --graphs <graphs.pt> --model_dir <result_dir> --num_locations <N>

Expected layout under model_dir:
    <model_dir>/<pipeline>/<label_short>/best_model.pt
    <model_dir>/<pipeline>/<label_short>/norm_params.pt
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


PIPELINES = ['stephy2', 'CBLV-CNN2', 'CBLV-GAT2']
LABEL_DIRS = {
    'R0': 'r0',
    'Recovery_Rate': 'rr',
    'Source_Sink_Score': 'sss',
    'Ancestral_State': 'as',
}
LABELS = list(LABEL_DIRS.keys())
STEPHY_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(STEPHY_ROOT / 'stephy2'))
from graph_loader import load_graphs


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

    if pipeline == 'CBLV-CNN2':
        return model_mod.CBLV_CNN, config
    else:  # stephy2 and CBLV-GAT2 both export CBLV_GAT
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
    """Run a forward pass and return flat and per-graph predictions/labels.

    Each pipeline has a different model call signature:
      - stephy2:   model(graph, cblv, aux, edge_feat)
      - CBLV-CNN2: model(cblv, aux)
      - CBLV-GAT2: model(graph, cblv, aux)
    """
    node_cblv = batched_g.ndata['cblv']
    node_aux = batched_g.ndata['aux']

    if pipeline == 'stephy2':
        preds = model(batched_g, node_cblv, node_aux, batched_g.edata['feat'])
    elif pipeline == 'CBLV-CNN2':
        preds = model(node_cblv, node_aux)
    else:  # CBLV-GAT2
        preds = model(batched_g, node_cblv, node_aux)

    labels = batched_g.ndata[label_name]
    splits = batched_g.batch_num_nodes().tolist()
    return preds, labels, torch.stack(preds.split(splits)), torch.stack(labels.split(splits))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, label_name, is_classification, pipeline):
    """Run inference over the full dataloader.

    Returns (preds, labels) as numpy arrays.
    For classification: per-graph argmax predictions.
    For regression: flat node-level predictions.
    """
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, label_name, is_classification,
                    label_norm, num_locations):
    """Compute evaluation metrics and build a predictions DataFrame.

    Classification: accuracy (overall and per-class).
    Regression: R2 and MSE after inverse-transforming to the original scale.
    """
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
    label_dir = LABEL_DIRS[label_name]
    model_path = model_dir / pipeline / label_dir
    is_classification = (label_name == 'Ancestral_State')

    best_model_file = model_path / 'best_model.pt'
    norm_file = model_path / 'norm_params.pt'
    if not best_model_file.exists():
        print(f"    SKIP: {best_model_file} not found")
        return None
    if not norm_file.exists():
        print(f"    SKIP: {norm_file} not found")
        return None

    # Deep copy so in-place normalizations (aux, edge, labels) applied for this
    # pipeline x label combination don't leak into subsequent combinations.
    # Each combination needs to start from the original un-normalized tensors.
    graphs = deepcopy(raw_graphs)

    # CBLV-GAT2: add self-loops (matches training)
    if pipeline == 'CBLV-GAT2':
        graphs = [(dgl.add_self_loop(g), *rest) for g, *rest in graphs]

    # Apply training-set normalization
    norm_params = torch.load(norm_file, weights_only=False)
    normalize_aux(graphs, norm_params['aux'])
    if pipeline == 'stephy2' and 'edge' in norm_params:
        normalize_edge(graphs, norm_params['edge'])

    label_norm = norm_params.get('label')
    if not is_classification and label_norm is not None:
        normalize_labels(graphs, label_name, label_norm)

    dataloader = GraphDataLoader([g for g, *_ in graphs], batch_size=32, shuffle=False)

    model_cls, config = get_model_and_config(pipeline)
    config['model']['subtree_width'] = graphs[0][0].ndata['cblv'].shape[2]
    model = model_cls(config['model'])
    model.load_state_dict(torch.load(best_model_file, weights_only=False))

    preds, labels = evaluate(model, dataloader, label_name, is_classification, pipeline)
    metrics, pred_df = compute_metrics(preds, labels, label_name, is_classification,
                                       label_norm, num_locations)

    save_dir = output_dir / pipeline / label_dir
    save_dir.mkdir(parents=True, exist_ok=True)
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

    for g, graph_id, locs, _ in raw_graphs:
        if g.num_nodes() != args.num_locations:
            raise ValueError(
                f"Graph {graph_id}: expected {args.num_locations} locations, "
                f"got {g.num_nodes()}."
            )

    summary_rows = []

    for pipeline in PIPELINES:
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline}")
        print(f"{'='*60}")

        for label_name in LABELS:
            is_cls = (label_name == 'Ancestral_State')
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
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'accuracy', 'metric_value': metrics['accuracy'],
                })
            else:
                print(f"    R2: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'r2', 'metric_value': metrics['r2'],
                })
                summary_rows.append({
                    'pipeline': pipeline, 'label': label_name,
                    'metric_name': 'mse', 'metric_value': metrics['mse'],
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
