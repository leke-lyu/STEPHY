#!/usr/bin/env python3
"""
Training script for CBLV-CNN (CNN + Aux Branch baseline, no graph structure).

Requires precomputed graphs from build_graphs.py.
Supports conformal prediction (CQR for regression, RAPS for classification)
when 'conformal_prediction' is True in config.
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from dgl.dataloading import GraphDataLoader

from model import CBLV_CNN, count_parameters
from config import get_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'stephy'))
from graph_loader import load_graphs
from conformal import (PinballLoss, split_data_cp,
                       calibrate_cqr, apply_cqr_test,
                       calibrate_raps, apply_raps_test,
                       compute_cp_metrics, save_cp_artifacts)


def parse_args():
    parser = argparse.ArgumentParser(description='CBLV-CNN: Train with phylogeny + aux data (no graph)')
    parser.add_argument('--graphs', required=True, help='Precomputed graphs file (from build_graphs.py)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_locations', type=int, required=True, help='Number of locations')
    parser.add_argument('--label', choices=['reg_r0', 'cls_r0', 'reg_rr', 'reg_sss', 'cls_sss', 'cls_as'], required=True, help='Label to predict (reg_=regression, cls_=classification)')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_aux_features(train_graphs, *other_graph_lists):
    """Normalize aux features: clamp(1e-8) -> log -> z-score (training set stats).

    Note: No edge normalization -- intentional for this CNN ablation baseline,
    which has no edges or graph structure.
    """
    train_aux = torch.cat([g.ndata['aux'] for g, *_ in train_graphs], dim=0)

    # Log transform (clamp to avoid log(0))
    train_aux = torch.log(train_aux.clamp(min=1e-8))
    mean = train_aux.mean(dim=0)
    std = train_aux.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, *other_graph_lists]:
        for g, *_ in graph_list:
            g.ndata['aux'] = (torch.log(g.ndata['aux'].clamp(min=1e-8)) - mean) / std

    return {'mean': mean, 'std': std}


def normalize_labels(train_graphs, *other_graph_lists, label_name):
    """Z-score normalize labels using training set statistics."""
    train_labels = torch.cat([g.ndata[label_name] for g, *_ in train_graphs])
    mean = train_labels.mean()
    std = train_labels.std()

    if std < 1e-8:
        std = torch.tensor(1.0)

    for graph_list in [train_graphs, *other_graph_lists]:
        for g, *_ in graph_list:
            g.ndata[label_name] = (g.ndata[label_name] - mean) / std

    return {'mean': mean, 'std': std}


def _forward_batch(model, batched_g, label_name):
    """Run model forward and reshape predictions/labels to per-graph."""
    node_cblv = batched_g.ndata['cblv']
    node_aux = batched_g.ndata['aux']

    predictions = model(node_cblv, node_aux)
    labels = batched_g.ndata[label_name]

    num_nodes_list = batched_g.batch_num_nodes().tolist()
    per_graph_preds = torch.stack(predictions.split(num_nodes_list))
    per_graph_labels = torch.stack(labels.split(num_nodes_list))

    return predictions, labels, per_graph_preds, per_graph_labels


def train_epoch(model, dataloader, optimizer, criterion, label_name, is_classification=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_graphs = 0

    for batched_g in dataloader:
        optimizer.zero_grad()
        _, _, per_graph_preds, per_graph_labels = _forward_batch(model, batched_g, label_name)

        if is_classification:
            loss = criterion(per_graph_preds, per_graph_labels.argmax(dim=1))
        else:
            loss = criterion(per_graph_preds, per_graph_labels)

        loss.backward()
        optimizer.step()

        batch_size = per_graph_preds.shape[0]
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs


def evaluate(model, dataloader, criterion, label_name, is_classification=False):
    """Evaluate model. Returns dict with 'loss', 'preds', and 'labels'."""
    model.eval()
    total_loss = 0
    total_graphs = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_g in dataloader:
            predictions, labels, per_graph_preds, per_graph_labels = _forward_batch(model, batched_g, label_name)

            if is_classification:
                targets = per_graph_labels.argmax(dim=1)
                loss = criterion(per_graph_preds, targets)
                all_preds.extend(per_graph_preds.argmax(dim=1).numpy())
                all_labels.extend(targets.numpy())
            else:
                loss = criterion(per_graph_preds, per_graph_labels)
                all_preds.extend(predictions.numpy())
                all_labels.extend(labels.numpy())

            batch_size = per_graph_preds.shape[0]
            total_loss += loss.item() * batch_size
            total_graphs += batch_size

    return {
        'loss': total_loss / total_graphs,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
    }


def main():
    args = parse_args()
    config = get_config()

    # Load graphs
    all_graphs = load_graphs(args.graphs)
    subtree_width = all_graphs[0][0].ndata['cblv'].shape[2]

    config['model']['subtree_width'] = subtree_width
    config['num_locations'] = args.num_locations

    label_name = args.label
    is_classification = label_name.startswith('cls_')
    use_cp = config['train'].get('conformal_prediction', False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config['train']['random_seed'])

    # Validate num_locations
    for g, graph_id, locs, _ in all_graphs:
        if g.num_nodes() != args.num_locations:
            raise ValueError(
                f"Graph {graph_id}: expected {args.num_locations} locations, got {g.num_nodes()}."
            )

    # Train/val/(cal)/test split
    train_ratio = config['train']['train_ratio']
    seed = config['train']['random_seed']
    if use_cp:
        train_graphs, val_graphs, cal_graphs, test_graphs = split_data_cp(
            all_graphs, train_ratio, seed)
    else:
        train_graphs, temp_graphs = train_test_split(
            all_graphs, test_size=1 - train_ratio, random_state=seed)
        val_graphs, test_graphs = train_test_split(
            temp_graphs, test_size=0.5, random_state=seed)
        cal_graphs = None

    # CQR: set model to output 3 quantiles for regression
    if use_cp and not is_classification:
        config['model']['num_outputs'] = len(config['train']['cqr_quantiles'])

    # Summary
    task_type = 'classification' if is_classification else 'regression'
    label_info = f"{label_name}({task_type})" if is_classification else f"{label_name}({task_type}) zscore"
    cp_info = f" CP={'CQR' if not is_classification else 'RAPS'}" if use_cp else ""
    split_info = (f"{len(train_graphs)}/{len(val_graphs)}/{len(cal_graphs)}/{len(test_graphs)}"
                  if cal_graphs else f"{len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    print(f"Data: {len(all_graphs)} graphs, {args.num_locations} locations, split {split_info}")
    print(f"Features: CBLV(4ch) {config['data']['cblv_scale']}")
    print(f"  Aux(mrca_depth, earliest_tip, latest_tip, avg_bl, n_tips) log+zscore")
    print(f"Label: {label_info}{cp_info}")

    # Normalize: aux -> label (no edge features)
    other_sets = [val_graphs, test_graphs] + ([cal_graphs] if cal_graphs else [])
    aux_norm = normalize_aux_features(train_graphs, *other_sets)

    label_norm = None
    if not is_classification:
        label_norm = normalize_labels(train_graphs, *other_sets, label_name=label_name)

    # Save normalization params
    torch.save({'aux': aux_norm, 'label': label_norm},
               output_dir / 'norm_params.pt')

    # Create DataLoaders
    train_g = [g for g, *_ in train_graphs]
    val_g = [g for g, *_ in val_graphs]
    test_g = [g for g, *_ in test_graphs]
    batch_size = config['train']['batch_size']
    train_loader = GraphDataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_g, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_g, batch_size=batch_size, shuffle=False)

    cal_loader = None
    if cal_graphs is not None:
        cal_g = [g for g, *_ in cal_graphs]
        cal_loader = GraphDataLoader(cal_g, batch_size=batch_size, shuffle=False)

    # Create model
    model = CBLV_CNN(config['model'])
    print(f"Model: {count_parameters(model):,} parameters")

    # Training setup
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    elif use_cp:
        criterion = PinballLoss(quantiles=config['train']['cqr_quantiles'])
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    num_epochs = config['train']['num_epochs']
    patience = config['train']['early_stopping_patience']

    # Training loop
    print("Training...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {'train_loss': [], 'val_loss': []}
    if is_classification:
        history['val_accuracy'] = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, label_name, is_classification)
        val_results = evaluate(model, val_loader, criterion, label_name, is_classification)
        val_loss = val_results['loss']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if is_classification:
            val_acc = accuracy_score(val_results['labels'], val_results['preds'])
            history['val_accuracy'].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if is_classification:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if is_classification:
            improved = val_acc > best_val_acc
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            if is_classification:
                best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if is_classification:
        print(f"Best epoch: {best_epoch+1} (val_acc={best_val_acc:.4f})")
    else:
        print(f"Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f})")

    # Load best model
    model.load_state_dict(best_state)

    # --- Conformal prediction calibration + test ---
    if use_cp:
        forward_fn = lambda m, bg: _forward_batch(m, bg, label_name)
        alpha = config['train']['cp_alpha']

        if is_classification:
            lambda_reg = config['train']['raps_lambda']
            k_reg = config['train']['raps_k_reg']

            print(f"\nRAPS calibration (alpha={alpha}, lambda={lambda_reg}, k={k_reg})...")
            cp_cal = calibrate_raps(model, cal_loader, label_name, alpha,
                                    lambda_reg, k_reg, forward_fn)
            print(f"  q_hat = {cp_cal['q_hat']:.4f}, n_cal = {len(cp_cal['cal_scores'])}")

            cp_test = apply_raps_test(model, test_loader, label_name, cp_cal['q_hat'],
                                      lambda_reg, k_reg, forward_fn)

            cp_metrics = compute_cp_metrics(cp_test, is_classification=True,
                                            num_locations=args.num_locations)
            print(f"  Coverage: {cp_metrics['empirical_coverage']:.4f}")
            print(f"  Mean set size: {cp_metrics['mean_set_size']:.2f}")
            print(f"  Singleton fraction: {cp_metrics['singleton_fraction']:.4f}")

            acc = accuracy_score(cp_test['true_classes'], cp_test['pred_classes'])
            print(f"  Point accuracy: {acc:.4f}")

            save_cp_artifacts(output_dir, cp_cal, cp_metrics, method='raps',
                              alpha=alpha, lambda_reg=lambda_reg, k_reg=k_reg)

            pred_df = pd.DataFrame({
                'true_ancestor': cp_test['true_classes'].astype(int),
                'pred_ancestor': cp_test['pred_classes'].astype(int),
                'prediction_set': [str(s) for s in cp_test['prediction_sets']],
                'set_size': cp_test['set_sizes'].astype(int),
                'covered': cp_test['covered'].astype(int),
            })
        else:
            print(f"\nCQR calibration (alpha={alpha})...")
            cp_cal = calibrate_cqr(model, cal_loader, label_name, alpha, forward_fn)
            print(f"  q_hat = {cp_cal['q_hat']:.4f} (z-space), n_cal = {len(cp_cal['cal_scores'])}")

            cp_test = apply_cqr_test(model, test_loader, label_name,
                                     cp_cal['q_hat'], label_norm, forward_fn)

            cp_metrics = compute_cp_metrics(cp_test, is_classification=False)
            print(f"  Coverage: {cp_metrics['empirical_coverage']:.4f}")
            print(f"  Mean interval width: {cp_metrics['mean_interval_width']:.4f}")

            r2 = r2_score(cp_test['true'], cp_test['point'])
            mse = mean_squared_error(cp_test['true'], cp_test['point'])
            print(f"  Point R2={r2:.4f}, MSE={mse:.4f}")

            save_cp_artifacts(output_dir, cp_cal, cp_metrics, method='cqr', alpha=alpha)

            pred_df = pd.DataFrame({
                f'true_{label_name}': cp_test['true'],
                f'pred_{label_name}': cp_test['point'],
                f'lower_{label_name}': cp_test['lower'],
                f'upper_{label_name}': cp_test['upper'],
                'interval_width': cp_test['width'],
                'covered': cp_test['covered'].astype(int),
            })

        _save_results(output_dir, best_state, history, pred_df)
    else:
        test_results = evaluate(model, test_loader, criterion, label_name, is_classification)
        test_preds = test_results['preds']
        test_labels = test_results['labels']

        pred_df = _print_metrics(test_preds, test_labels, is_classification, label_name,
                                  label_norm, args.num_locations)
        _save_results(output_dir, best_state, history, pred_df)


def _print_metrics(test_preds, test_labels, is_classification, label_name,
                   label_norm, num_locations):
    """Print test metrics and return prediction DataFrame."""
    if is_classification:
        acc = accuracy_score(test_labels, test_preds)
        print(f"Test: Accuracy={acc:.4f}")

        for c in range(num_locations):
            mask = test_labels == c
            if mask.sum() > 0:
                class_acc = (test_preds[mask] == c).mean()
                print(f"  Class {c}: {mask.sum()} samples, accuracy={class_acc:.4f}")

        return pd.DataFrame({
            'true_ancestor': test_labels.astype(int),
            'pred_ancestor': test_preds.astype(int),
        })
    else:
        if label_norm is not None:
            label_mean = label_norm['mean'].item()
            label_std = label_norm['std'].item()
            test_preds = test_preds * label_std + label_mean
            test_labels = test_labels * label_std + label_mean

        r2 = r2_score(test_labels, test_preds)
        mse = mean_squared_error(test_labels, test_preds)
        print(f"Test: R2={r2:.4f}, MSE={mse:.4f}")

        return pd.DataFrame({
            f'true_{label_name}': test_labels,
            f'pred_{label_name}': test_preds,
        })


def _save_results(output_dir, best_state, history, pred_df):
    """Save model, training history, and predictions."""
    torch.save(best_state, output_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
