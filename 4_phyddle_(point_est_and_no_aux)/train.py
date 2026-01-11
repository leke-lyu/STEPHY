#!/usr/bin/env python3
"""
Simplified training script for phylogenetic R0 estimation.
No auxiliary data, point estimates only.

Usage:
    python3 train.py
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from config import args
from model import SimplifiedPhyloNet, PhyloDataset


def load_hdf5_data(filepath):
    """Load data from phyddle-formatted HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        phy_data = f['phy_data'][:]
        labels = f['labels'][:]
        idx = f['idx'][:]
        label_names = [s.decode() for s in f['label_names'][0, :]]
    return phy_data, labels, idx, label_names


def reshape_phy_data(phy_data_flat, tree_width, num_channels):
    """Reshape flattened phy_data to (samples, channels, tree_width)."""
    n_samples = phy_data_flat.shape[0]
    phy_data = phy_data_flat.reshape(n_samples, tree_width, num_channels)
    phy_data = np.transpose(phy_data, (0, 2, 1))
    return phy_data


def normalize_data(train_data, val_data=None, test_data=None):
    """Normalize data using training set statistics."""
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std == 0] = 1

    train_norm = (train_data - mean) / std
    results = [train_norm, mean, std]

    if val_data is not None:
        results.append((val_data - mean) / std)
    if test_data is not None:
        results.append((test_data - mean) / std)

    return results


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for phy_batch, label_batch in dataloader:
        phy_batch = phy_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        outputs = model(phy_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for phy_batch, label_batch in dataloader:
            phy_batch = phy_batch.to(device)
            label_batch = label_batch.to(device)

            outputs = model(phy_batch)
            loss = criterion(outputs, label_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

    return total_loss / n_batches, np.vstack(all_preds), np.vstack(all_labels)


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set random seed for reproducibility
    random_seed = args.get('random_seed', 42)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    num_proc = args.get('num_proc', -1)
    if num_proc <= 0:
        num_proc = max(1, cpu_count() + num_proc)
    torch.set_num_threads(num_proc)
    print(f"CPU threads: {num_proc}")

    os.makedirs(args['output_dir'], exist_ok=True)

    # Load data
    prefix = args['prefix']
    format_dir = './format_output'

    if not os.path.exists(format_dir):
        print("Error: format_output directory not found.")
        print("Please run: python3 -m phyddle -c config_format.py -s F")
        sys.exit(1)

    train_hdf5 = os.path.join(format_dir, f"{prefix}.train.hdf5")
    test_hdf5 = os.path.join(format_dir, f"{prefix}.test.hdf5")

    train_phy_flat, train_labels, train_idx, label_names = load_hdf5_data(train_hdf5)
    test_phy_flat, test_labels, test_idx, _ = load_hdf5_data(test_hdf5)

    print(f"Data: train={len(train_labels)}, test={len(test_labels)}")

    # Calculate data dimensions
    tree_width = args['tree_width']
    total_length = train_phy_flat.shape[1]
    num_channels = total_length // tree_width

    if total_length % tree_width != 0:
        print(f"Warning: Data length {total_length} not divisible by tree_width {tree_width}")
        for nc in [20, 19, 18, 17, 16]:
            if total_length % nc == 0:
                tree_width = total_length // nc
                num_channels = nc
                print(f"Auto-detected: tree_width={tree_width}, num_channels={num_channels}")
                break

    print(f"Data: tree_width={tree_width}, num_channels={num_channels}")

    # Reshape data
    train_phy = reshape_phy_data(train_phy_flat, tree_width, num_channels)
    test_phy = reshape_phy_data(test_phy_flat, tree_width, num_channels)

    # Split training into train/val
    n_train = len(train_labels)
    n_val = int(n_train * args['prop_val'])
    indices = np.random.permutation(n_train)

    train_phy_split = train_phy[indices[:-n_val]]
    train_labels_split = train_labels[indices[:-n_val]]
    val_phy = train_phy[indices[-n_val:]]
    val_labels = train_labels[indices[-n_val:]]

    print(f"Splits: train={len(train_labels_split)}, val={len(val_labels)}")

    # Normalize labels
    train_labels_norm, label_mean, label_std, val_labels_norm, test_labels_norm = \
        normalize_data(train_labels_split, val_labels, test_labels)

    # Create datasets and dataloaders
    train_dataset = PhyloDataset(train_phy_split, train_labels_norm)
    val_dataset = PhyloDataset(val_phy, val_labels_norm)
    test_dataset = PhyloDataset(test_phy, test_labels_norm)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0)

    # Create model
    model = SimplifiedPhyloNet(
        input_channels=num_channels,
        num_outputs=args['num_params'],
        args=args
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    print("\nTraining...")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(args['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args['output_dir'], 'best_model.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args['num_early_stop']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest epoch: {best_epoch+1} (val_loss: {best_val_loss:.4f})")

    # Evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args['output_dir'], 'best_model.pt')))
    test_loss, test_preds_norm, test_labels_norm_eval = evaluate(model, test_loader, criterion, device)

    # Denormalize predictions
    test_preds = test_preds_norm * label_std + label_mean
    test_labels_denorm = test_labels_norm_eval * label_std + label_mean

    test_mse_orig = np.mean((test_preds - test_labels_denorm) ** 2)
    print(f"Test MSE: {test_mse_orig:.4f}")

    # Save results
    pd.DataFrame(history).to_csv(os.path.join(args['output_dir'], 'training_history.csv'), index=False)

    pred_df = pd.DataFrame(test_preds, columns=[f'{name}_pred' for name in label_names])
    true_df = pd.DataFrame(test_labels_denorm, columns=[f'{name}_true' for name in label_names])
    pd.concat([true_df, pred_df], axis=1).to_csv(os.path.join(args['output_dir'], 'test_predictions.csv'), index=False)

    pd.DataFrame({
        'param': label_names,
        'mean': label_mean.flatten(),
        'std': label_std.flatten()
    }).to_csv(os.path.join(args['output_dir'], 'normalization_params.csv'), index=False)

    print(f"\nSaved to {args['output_dir']}/: best_model.pt, training_history.csv, test_predictions.csv, normalization_params.csv")
    print("Done!")


if __name__ == '__main__':
    main()
