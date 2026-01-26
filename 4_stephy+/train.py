#!/usr/bin/env python3
"""
Training script for STEPHY+ (Phylogeny + Epidemiological Data).

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
from sklearn.metrics import r2_score, mean_squared_error
from dgl.dataloading import GraphDataLoader

from model import CBLV_GAT, count_parameters
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='STEPHY+: Train with phylogeny + epi data')
    parser.add_argument('--graphs', required=True, help='Precomputed graphs file (from build_graphs.py)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_locations', type=int, required=True, help='Number of locations')
    parser.add_argument('--label', choices=['R0', 'Source_Sink_Score'], required=True, help='Label to predict')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def apply_epi_log_transform(graphs):
    """Apply log transform to epi features (all positive, no +1 needed)."""
    for g, *_ in graphs:
        epi = g.ndata['epi']
        epi = torch.clamp(epi, min=1e-8)
        g.ndata['epi'] = torch.log(epi)


def normalize_epi_features(train_graphs, val_graphs, test_graphs):
    """Z-score normalize epi features using training set statistics."""
    train_epi = torch.cat([g.ndata['epi'] for g, *_ in train_graphs], dim=0)

    mean = train_epi.mean(dim=0)
    std = train_epi.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['epi'] = (g.ndata['epi'] - mean) / std

    return {'mean': mean, 'std': std}


def normalize_edge_features(train_graphs, val_graphs, test_graphs):
    """Z-score normalize edge features using training set statistics."""
    train_edge = torch.cat([g.edata['feat'] for g, *_ in train_graphs], dim=0)

    mean = train_edge.mean(dim=0)
    std = train_edge.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.edata['feat'] = (g.edata['feat'] - mean) / std

    return {'mean': mean, 'std': std}


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


def train_epoch(model, dataloader, optimizer, criterion, device, label_name):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_nodes = 0

    for batched_g in dataloader:
        batched_g = batched_g.to(device)
        node_cblv = batched_g.ndata['cblv']
        edge_feat = batched_g.edata['feat']

        optimizer.zero_grad()
        predictions = model(batched_g, node_cblv, edge_feat)
        labels = batched_g.ndata[label_name]

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        n_nodes = batched_g.num_nodes()
        total_loss += loss.item() * n_nodes
        total_nodes += n_nodes

    return total_loss / total_nodes


def evaluate(model, dataloader, criterion, device, label_name):
    """Evaluate model. Returns loss and predictions."""
    model.eval()
    total_loss = 0
    total_nodes = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_g in dataloader:
            batched_g = batched_g.to(device)
            node_cblv = batched_g.ndata['cblv']
            edge_feat = batched_g.edata['feat']

            predictions = model(batched_g, node_cblv, edge_feat)
            labels = batched_g.ndata[label_name]

            loss = criterion(predictions, labels)

            n_nodes = batched_g.num_nodes()
            total_loss += loss.item() * n_nodes
            total_nodes += n_nodes

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / total_nodes,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
    }


def main():
    args = parse_args()
    config = get_config()

    # Load graphs
    all_graphs = torch.load(args.graphs)
    subtree_width = all_graphs[0][0].ndata['cblv'].shape[2]

    config['model']['subtree_width'] = subtree_width
    config['num_locations'] = args.num_locations

    label_name = args.label
    data_cfg = config['data']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
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
    print(f"Data: {len(all_graphs)} graphs, {args.num_locations} locations, split {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    print(f"Features: CBLV [0,1], Epi log={data_cfg.get('epi_log', True)} {data_cfg.get('epi_norm', 'zscore')}, Edge {data_cfg.get('edge_norm', 'zscore')}")
    print(f"Label: {label_name} {data_cfg.get('label_norm', 'zscore')}")

    # Apply transforms based on config
    if data_cfg.get('epi_log', True):
        apply_epi_log_transform(train_graphs)
        apply_epi_log_transform(val_graphs)
        apply_epi_log_transform(test_graphs)

    # Normalize features based on config
    epi_norm = None
    if data_cfg.get('epi_norm', 'zscore') == 'zscore':
        epi_norm = normalize_epi_features(train_graphs, val_graphs, test_graphs)

    edge_norm = None
    if data_cfg.get('edge_norm', 'zscore') == 'zscore':
        edge_norm = normalize_edge_features(train_graphs, val_graphs, test_graphs)

    label_norm = None
    if data_cfg.get('label_norm', 'zscore') == 'zscore':
        label_norm = normalize_labels(train_graphs, val_graphs, test_graphs, label_name)

    # Save normalization params
    torch.save({'epi': epi_norm, 'edge': edge_norm, 'label': label_norm, 'config': data_cfg},
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
    model = CBLV_GAT(config['model']).to(device)
    print(f"Model: {count_parameters(model):,} parameters")

    # Training setup
    criterion = nn.MSELoss()
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

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, label_name)
        val_results = evaluate(model, val_loader, criterion, device, label_name)
        val_loss = val_results['loss']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
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
    test_results = evaluate(model, test_loader, criterion, device, label_name)

    # Denormalize predictions and labels
    test_preds = test_results['preds']
    test_labels = test_results['labels']
    if label_norm is not None:
        label_mean = label_norm['mean'].item()
        label_std = label_norm['std'].item()
        test_preds = test_preds * label_std + label_mean
        test_labels = test_labels * label_std + label_mean

    # Compute metrics
    r2 = r2_score(test_labels, test_preds)
    mse = mean_squared_error(test_labels, test_preds)
    print(f"Test: R2={r2:.4f}, MSE={mse:.4f}")

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    pd.DataFrame({
        f'true_{label_name}': test_labels,
        f'pred_{label_name}': test_preds,
    }).to_csv(output_dir / 'test_predictions.csv', index=False)
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
