#!/usr/bin/env python3
"""
CBLV-CNN Model: wider CNN + aux branch, no graph structure.

CNN (192) + Aux (64) = 256-dim node embedding -> classifier (no GAT).
Ablation of stephy: each node predicted independently.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'stephy'))
from model_core import _ACTIVATIONS, _kaiming_init, CBLVConvEncoder, AuxBranch, count_parameters


class CBLV_CNN(nn.Module):
    """
    CNN + aux branch baseline for single-task prediction (no graph).

    Architecture:
        1. CBLVConvEncoder: CBLV (4, W) -> 192-dim (wider than stephy's 96)
        2. AuxBranch: 5-dim stats -> 64-dim (wider than stephy's 32)
        3. Concat -> 256-dim node embedding
        4. Classifier: 256 -> 128 -> 64 -> 32 -> 1
    """

    def __init__(self, args):
        super().__init__()
        self.cnn_encoder = CBLVConvEncoder(args)
        self.aux_branch = AuxBranch(args)
        node_dim = self.cnn_encoder.output_dim + self.aux_branch.output_dim

        self.act_fn = _ACTIVATIONS.get(args['activation_func'], torch.relu)

        self.classifier = nn.ModuleList()
        in_features = node_dim
        for out_features in args['lbl_channel']:
            self.classifier.append(nn.Linear(in_features, out_features))
            in_features = out_features
        num_outputs = args.get('num_outputs', 1)
        self.classifier.append(nn.Linear(in_features, num_outputs))
        _kaiming_init(self.classifier)

    def forward(self, node_cblv, node_aux):
        """(CBLV, aux) -> (N,) or (N, Q) per-node predictions."""
        h_cnn = self.cnn_encoder(node_cblv)
        h_aux = self.aux_branch(node_aux)
        h = torch.cat([h_cnn, h_aux], dim=1)

        for layer in self.classifier[:-1]:
            h = self.act_fn(layer(h))
        out = self.classifier[-1](h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
