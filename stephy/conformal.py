#!/usr/bin/env python3
"""
Conformal prediction for STEPHY pipelines.

Shared module used by all 3 pipelines (stephy, CBLV-CNN, CBLV-GAT).
Provides CQR (Conformalized Quantile Regression) for regression tasks
and RAPS (Regularized Adaptive Prediction Sets) for classification tasks.

Usage in train.py:
    from conformal import (PinballLoss, split_data_cp,
                           calibrate_cqr, apply_cqr_test,
                           calibrate_raps, apply_raps_test,
                           compute_cp_metrics)
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def split_data_cp(all_graphs, train_ratio, seed):
    """Split data into train/val/cal/test (remaining 20% split 3 ways).

    Returns:
        (train_graphs, val_graphs, cal_graphs, test_graphs)
    """
    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=1 - train_ratio, random_state=seed
    )
    val_graphs, temp2 = train_test_split(
        temp_graphs, test_size=2 / 3, random_state=seed
    )
    cal_graphs, test_graphs = train_test_split(
        temp2, test_size=0.5, random_state=seed
    )
    return train_graphs, val_graphs, cal_graphs, test_graphs


# ---------------------------------------------------------------------------
# CQR: Pinball Loss
# ---------------------------------------------------------------------------

class PinballLoss(nn.Module):
    """Pinball (quantile) loss for CQR training.

    For each quantile tau:
        L = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))

    Args:
        quantiles: list of quantile levels, e.g. [0.05, 0.5, 0.95]
    """

    def __init__(self, quantiles=(0.05, 0.5, 0.95)):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds, targets):
        """
        Args:
            preds:   (batch, N, Q) where Q = len(quantiles)
            targets: (batch, N)
        Returns:
            scalar loss averaged over all samples and quantiles
        """
        # targets -> (batch, N, 1) for broadcasting
        targets = targets.unsqueeze(-1)
        errors = targets - preds  # (batch, N, Q)
        # quantiles shape: (Q,) -> broadcast with (batch, N, Q)
        loss = torch.max(self.quantiles * errors, (self.quantiles - 1) * errors)
        return loss.mean()


# ---------------------------------------------------------------------------
# CQR: Calibration
# ---------------------------------------------------------------------------

def calibrate_cqr(model, cal_loader, label_name, alpha, forward_fn):
    """Compute CQR calibration threshold on the calibration set.

    All computation is in z-space (normalized labels).

    Args:
        model: trained model (outputs 3 quantiles per node)
        cal_loader: DataLoader for calibration set
        label_name: label key in g.ndata
        alpha: miscoverage rate (e.g. 0.1 for 90% coverage)
        forward_fn: callable(model, batched_g) -> (flat_preds, flat_labels,
                    per_graph_preds, per_graph_labels)

    Returns:
        dict with 'q_hat' (scalar) and 'cal_scores' (np.ndarray)
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batched_g in cal_loader:
            flat_preds, flat_labels, _, _ = forward_fn(model, batched_g)
            # flat_preds: (N_total, 3) for CQR
            # flat_labels: (N_total,)
            q_lo = flat_preds[:, 0]
            q_hi = flat_preds[:, 2]

            # Handle quantile crossing: ensure q_lo <= q_hi
            q_lo_safe = torch.min(q_lo, q_hi)
            q_hi_safe = torch.max(q_lo, q_hi)

            scores = torch.max(q_lo_safe - flat_labels, flat_labels - q_hi_safe)
            all_scores.extend(scores.numpy())

    cal_scores = np.array(all_scores)
    n = len(cal_scores)
    # Conformal quantile: ceil((n+1)(1-alpha)) / n
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))

    return {'q_hat': q_hat, 'cal_scores': cal_scores}


def apply_cqr_test(model, test_loader, label_name, q_hat, label_norm, forward_fn):
    """Apply CQR to test set. Returns results in TRUE scale.

    Args:
        model: trained CQR model
        test_loader: DataLoader for test set
        label_name: label key
        q_hat: calibration threshold (z-space)
        label_norm: {'mean': tensor, 'std': tensor} or None
        forward_fn: callable(model, batched_g) -> (flat_preds, flat_labels, ...)

    Returns:
        dict with arrays: 'true', 'point', 'lower', 'upper', 'width', 'covered'
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_g in test_loader:
            flat_preds, flat_labels, _, _ = forward_fn(model, batched_g)
            all_preds.append(flat_preds.numpy())
            all_labels.append(flat_labels.numpy())

    preds = np.concatenate(all_preds)    # (N, 3)
    labels = np.concatenate(all_labels)  # (N,)

    q_lo = preds[:, 0]
    q_mid = preds[:, 1]
    q_hi = preds[:, 2]

    # Handle quantile crossing
    q_lo, q_hi = np.minimum(q_lo, q_hi), np.maximum(q_lo, q_hi)

    # Conformalized intervals in z-space
    lower_z = q_lo - q_hat
    upper_z = q_hi + q_hat

    # Inverse transform to true scale
    if label_norm is not None:
        lm = label_norm['mean'].item()
        ls = label_norm['std'].item()
        lower = lower_z * ls + lm
        upper = upper_z * ls + lm
        point = q_mid * ls + lm
        true = labels * ls + lm
    else:
        lower = lower_z
        upper = upper_z
        point = q_mid
        true = labels

    width = upper - lower
    covered = ((true >= lower) & (true <= upper)).astype(float)

    return {
        'true': true,
        'point': point,
        'lower': lower,
        'upper': upper,
        'width': width,
        'covered': covered,
    }


# ---------------------------------------------------------------------------
# RAPS: Calibration
# ---------------------------------------------------------------------------

def calibrate_raps(model, cal_loader, label_name, alpha, lambda_reg, k_reg,
                   forward_fn):
    """Compute RAPS calibration threshold on the calibration set.

    Conformity score for a sample whose true class sits at rank r (1-indexed)
    in the descending-probability ordering:

        E = sum_{k=1..r} p_(k) + lambda_reg * max(r - k_reg, 0)

    The penalty enters once, at the true class's rank.

    Args:
        model: trained classification model (outputs logits)
        cal_loader: DataLoader for calibration set
        label_name: label key in g.ndata
        alpha: miscoverage rate
        lambda_reg: RAPS regularization strength
        k_reg: number of "free" classes before penalty starts
        forward_fn: callable(model, batched_g) -> (flat_preds, flat_labels,
                    per_graph_preds, per_graph_labels)

    Returns:
        dict with 'q_hat' (scalar) and 'cal_scores' (np.ndarray)
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batched_g in cal_loader:
            _, _, pg_preds, pg_labels = forward_fn(model, batched_g)
            # pg_preds: (batch, num_locations) — raw logits
            # pg_labels: (batch, num_locations) — one-hot

            probs = F.softmax(pg_preds, dim=1)  # (batch, num_locations)
            true_classes = pg_labels.argmax(dim=1)  # (batch,)

            for i in range(probs.shape[0]):
                p = probs[i]  # (num_locations,)
                true_cls = true_classes[i].item()

                # Sort by descending probability
                sorted_probs, sorted_idx = torch.sort(p, descending=True)

                # Find rank of true class (0-indexed)
                true_rank = (sorted_idx == true_cls).nonzero(as_tuple=True)[0].item()

                # RAPS score: cumulative prob up to and including the true class,
                # plus a single rank penalty keyed to the true class's rank.
                # The penalty is applied once (not once per rank) so it grows
                # linearly in rank, per Angelopoulos et al.
                prob_sum = sorted_probs[:true_rank + 1].sum().item()
                score = prob_sum + lambda_reg * max(true_rank + 1 - k_reg, 0)

                all_scores.append(score)

    cal_scores = np.array(all_scores)
    n = len(cal_scores)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))

    return {'q_hat': q_hat, 'cal_scores': cal_scores}


def apply_raps_test(model, test_loader, label_name, q_hat, lambda_reg, k_reg,
                    forward_fn):
    """Apply RAPS to test set to produce prediction sets.

    Adds classes in descending probability order, stopping once the score
    `sum_{k=1..j} p_(k) + lambda_reg * max(j - k_reg, 0)` reaches `q_hat` —
    the same single-penalty form used in `calibrate_raps`.

    Args:
        model: trained classification model
        test_loader: DataLoader for test set
        label_name: label key
        q_hat: RAPS calibration threshold
        lambda_reg: RAPS regularization strength
        k_reg: number of "free" classes
        forward_fn: callable(model, batched_g) -> (flat_preds, flat_labels,
                    per_graph_preds, per_graph_labels)

    Returns:
        dict with: 'true_classes', 'pred_classes', 'prediction_sets',
                   'set_sizes', 'covered'
    """
    model.eval()
    all_true = []
    all_pred = []
    all_sets = []
    all_sizes = []
    all_covered = []

    with torch.no_grad():
        for batched_g in test_loader:
            _, _, pg_preds, pg_labels = forward_fn(model, batched_g)
            probs = F.softmax(pg_preds, dim=1)
            true_classes = pg_labels.argmax(dim=1)
            pred_classes = pg_preds.argmax(dim=1)

            for i in range(probs.shape[0]):
                p = probs[i]
                true_cls = true_classes[i].item()

                sorted_probs, sorted_idx = torch.sort(p, descending=True)

                # Build prediction set. The rank penalty is recomputed fresh at
                # each rank rather than accumulated, mirroring calibration.
                pred_set = []
                prob_sum = 0.0
                for j in range(len(sorted_probs)):
                    prob_sum += sorted_probs[j].item()
                    pred_set.append(sorted_idx[j].item())
                    score = prob_sum + lambda_reg * max(j + 1 - k_reg, 0)
                    if score >= q_hat:
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

def compute_cp_metrics(cp_results, is_classification, num_locations=None):
    """Compute conformal prediction diagnostic metrics.

    Args:
        cp_results: dict from apply_cqr_test or apply_raps_test
        is_classification: True for RAPS, False for CQR
        num_locations: number of locations (for per-class metrics)

    Returns:
        dict of metrics
    """
    metrics = {}

    if is_classification:
        covered = cp_results['covered']
        sizes = cp_results['set_sizes']
        true_cls = cp_results['true_classes']

        metrics['empirical_coverage'] = float(covered.mean())
        metrics['mean_set_size'] = float(sizes.mean())
        metrics['median_set_size'] = float(np.median(sizes))
        metrics['singleton_fraction'] = float((sizes == 1).mean())
        metrics['n_test'] = int(len(covered))

        # Set size distribution
        if num_locations is not None:
            for s in range(1, num_locations + 1):
                frac = float((sizes == s).mean())
                if frac > 0:
                    metrics[f'set_size_{s}_fraction'] = frac

        # Per-class conditional coverage
        if num_locations is not None:
            for c in range(num_locations):
                mask = true_cls == c
                if mask.sum() > 0:
                    metrics[f'class_{c}_coverage'] = float(covered[mask].mean())
                    metrics[f'class_{c}_mean_set_size'] = float(sizes[mask].mean())
    else:
        covered = cp_results['covered']
        width = cp_results['width']
        true = cp_results['true']

        metrics['empirical_coverage'] = float(covered.mean())
        metrics['mean_interval_width'] = float(width.mean())
        metrics['median_interval_width'] = float(np.median(width))
        metrics['n_test'] = int(len(covered))

    return metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_cp_artifacts(output_dir, cp_cal, cp_metrics, method, alpha,
                      lambda_reg=None, k_reg=None):
    """Save conformal prediction calibration and metrics.

    Args:
        output_dir: Path to output directory
        cp_cal: dict from calibrate_cqr or calibrate_raps
        cp_metrics: dict from compute_cp_metrics
        method: 'cqr' or 'raps'
        alpha: miscoverage rate
        lambda_reg: RAPS lambda (None for CQR)
        k_reg: RAPS k (None for CQR)
    """
    cal_data = {
        'q_hat': cp_cal['q_hat'],
        'cal_scores': cp_cal['cal_scores'],
        'alpha': alpha,
        'method': method,
    }
    if method == 'raps':
        cal_data['lambda_reg'] = lambda_reg
        cal_data['k_reg'] = k_reg

    torch.save(cal_data, output_dir / 'cp_calibration.pt')

    with open(output_dir / 'cp_metrics.json', 'w') as f:
        json.dump(cp_metrics, f, indent=2)


def load_cp_calibration(model_path):
    """Load saved CP calibration from a model directory.

    Returns:
        dict with 'q_hat', 'method', 'alpha', etc. or None if not found.
    """
    cp_file = model_path / 'cp_calibration.pt'
    if not cp_file.exists():
        return None
    return torch.load(cp_file, weights_only=False)
