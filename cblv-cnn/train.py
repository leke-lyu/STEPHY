#!/usr/bin/env python3
"""
Training script for CBLV-CNN Baseline (No Graph Structure).

Requires precomputed graphs from build_graphs.py.
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description='CBLV-CNN: Train with phylogeny data (no graph structure)')
    parser.add_argument('--graphs', required=True, help='Precomputed graphs file (from build_graphs.py)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_locations', type=int, required=True, help='Number of locations')
    parser.add_argument('--label', choices=['R0', 'Source_Sink_Score', 'Recovery_Rate', 'Ancestral_State'], required=True, help='Label to predict')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_labels(train_graphs, val_graphs, test_graphs, label_name):
    """Z-score normalize labels using training set statistics."""
    train_labels = torch.cat([g.ndata[label_name] for g, *_ in train_graphs])
    mean = train_labels.mean()
    std = train_labels.std()

    if std < 1e-8:
        std = torch.tensor(1.0)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata[label_name] = (g.ndata[label_name] - mean) / std

    return {'mean': mean, 'std': std}


def train_epoch(model, dataloader, optimizer, criterion, label_name, is_classification=False):
    """
    Train for one epoch.

    Loss is computed per-graph: the flat per-node predictions are reshaped to
    (batch, N) where N = num_locations. For classification, argmax of the
    one-hot labels gives integer targets for CrossEntropyLoss. For regression,
    MSELoss over (batch, N) is numerically identical to per-node MSE since all
    graphs have the same N.
    """
    model.train()
    total_loss = 0
    total_graphs = 0

    for batched_g in dataloader:
        node_cblv = batched_g.ndata['cblv']

        optimizer.zero_grad()
        predictions = model(node_cblv)
        labels = batched_g.ndata[label_name]

        num_nodes_list = batched_g.batch_num_nodes().tolist()
        per_graph_preds = torch.stack(predictions.split(num_nodes_list))
        per_graph_labels = torch.stack(labels.split(num_nodes_list))

        if is_classification:
            targets = per_graph_labels.argmax(dim=1)
            loss = criterion(per_graph_preds, targets)
        else:
            loss = criterion(per_graph_preds, per_graph_labels)

        loss.backward()
        optimizer.step()

        batch_size = len(num_nodes_list)
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs


def evaluate(model, dataloader, criterion, label_name, is_classification=False):
    """
    Evaluate model. Returns dict with 'loss', 'preds', and 'labels'.

    For classification: preds/labels are per-graph integer class IDs.
    For regression: preds/labels are per-node scalar values.
    """
    model.eval()
    total_loss = 0
    total_graphs = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_g in dataloader:
            node_cblv = batched_g.ndata['cblv']

            predictions = model(node_cblv)
            labels = batched_g.ndata[label_name]

            num_nodes_list = batched_g.batch_num_nodes().tolist()
            per_graph_preds = torch.stack(predictions.split(num_nodes_list))
            per_graph_labels = torch.stack(labels.split(num_nodes_list))

            if is_classification:
                targets = per_graph_labels.argmax(dim=1)
                loss = criterion(per_graph_preds, targets)
                all_preds.extend(per_graph_preds.argmax(dim=1).numpy())
                all_labels.extend(targets.numpy())
            else:
                loss = criterion(per_graph_preds, per_graph_labels)
                all_preds.extend(predictions.numpy())
                all_labels.extend(labels.numpy())

            batch_size = len(num_nodes_list)
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
    all_graphs = torch.load(args.graphs, weights_only=False)
    subtree_width = all_graphs[0][0].ndata['cblv'].shape[2]

    config['model']['subtree_width'] = subtree_width
    config['num_locations'] = args.num_locations

    label_name = args.label
    is_classification = (label_name == 'Ancestral_State')
    data_cfg = config['data']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config['train']['random_seed'])

    # Validate num_locations
    for g, graph_id, locs, _ in all_graphs:
        if g.num_nodes() != args.num_locations:
            raise ValueError(
                f"Graph {graph_id}: expected {args.num_locations} locations, got {g.num_nodes()}."
            )

    # Train/val/test split
    train_ratio = config['train']['train_ratio']
    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=1-train_ratio, random_state=config['train']['random_seed']
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.5, random_state=config['train']['random_seed']
    )

    # Summary
    task_type = 'classification' if is_classification else 'regression'
    print(f"Data: {len(all_graphs)} graphs, {args.num_locations} locations, split {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    print(f"Features: CBLV [0,1] (no edge features)")
    print(f"Label: {label_name} ({task_type})")

    # Normalize labels (no edge normalization for this model)
    label_norm = None
    if not is_classification and data_cfg.get('label_norm', 'zscore') == 'zscore':
        label_norm = normalize_labels(train_graphs, val_graphs, test_graphs, label_name)

    # Save normalization params
    torch.save({'edge': None, 'label': label_norm, 'config': data_cfg},
               output_dir / 'norm_params.pt')

    # Create DataLoaders
    train_g = [g for g, *_ in train_graphs]
    val_g = [g for g, *_ in val_graphs]
    test_g = [g for g, *_ in test_graphs]
    batch_size = config['train']['batch_size']
    train_loader = GraphDataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_g, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_g, batch_size=batch_size, shuffle=False)

    # Create model
    model = CBLV_CNN(config['model'])
    print(f"Model: {count_parameters(model):,} parameters")

    # Training setup
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    num_epochs = config['train']['num_epochs']
    patience = config['train']['early_stopping_patience']

    # Training loop
    print("Training...")
    best_val_loss = float('inf')
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f})")

    # Evaluate on test set
    model.load_state_dict(best_state)
    test_results = evaluate(model, test_loader, criterion, label_name, is_classification)

    test_preds = test_results['preds']
    test_labels = test_results['labels']

    # Compute metrics and build prediction DataFrame
    if is_classification:
        acc = accuracy_score(test_labels, test_preds)
        print(f"Test: Accuracy={acc:.4f}")

        for c in range(args.num_locations):
            mask = test_labels == c
            if mask.sum() > 0:
                class_acc = (test_preds[mask] == c).mean()
                print(f"  Class {c}: {mask.sum()} samples, accuracy={class_acc:.4f}")

        pred_df = pd.DataFrame({
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

        pred_df = pd.DataFrame({
            f'true_{label_name}': test_labels,
            f'pred_{label_name}': test_preds,
        })

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
