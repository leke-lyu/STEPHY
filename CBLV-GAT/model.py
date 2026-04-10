#!/usr/bin/env python3
"""
CBLV-GAT Model: CNN encoder + standard GATConv (node-based attention).

CNN (96) + Aux (32) = 128-dim node embedding -> GATConv (4 heads x 64) -> 256-dim -> classifier.
Ablation of stephy: attention from node embeddings only, no DTW edge features.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from dgl.nn import GATConv

sys.path.append(str(Path(__file__).resolve().parent.parent / 'stephy'))
from model_core import _ACTIVATIONS, _kaiming_init, CBLVConvEncoder, AuxBranch, count_parameters


class CBLV_GAT(nn.Module):
    """
    CNN + aux + standard GATConv for single-task prediction.

    Architecture:
        1. CBLVConvEncoder: CBLV (4, W) -> 96-dim
        2. AuxBranch: 5-dim stats -> 32-dim
        3. Concat -> 128-dim node embedding
        4. GATConv (4 heads x 64, node-based attention) -> 256-dim
        5. Classifier: 256 -> 128 -> 64 -> 32 -> 1
    """

    def __init__(self, args):
        super().__init__()
        self.cnn_encoder = CBLVConvEncoder(args)
        self.aux_branch = AuxBranch(args)
        node_dim = self.cnn_encoder.output_dim + self.aux_branch.output_dim

        num_heads = args['gat_num_heads']
        gat_out_dim = args['gat_out_dim']
        self.gat_layer = GATConv(node_dim, gat_out_dim, num_heads)

        gat_output_dim = num_heads * gat_out_dim
        self.act_fn = _ACTIVATIONS.get(args['activation_func'], torch.relu)

        self.classifier = nn.ModuleList()
        in_features = gat_output_dim
        for out_features in args['lbl_channel']:
            self.classifier.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.classifier.append(nn.Linear(in_features, 1))
        _kaiming_init(self.classifier)

    def forward(self, g, node_cblv, node_aux):
        """(g, CBLV, aux) -> (N,) per-node predictions."""
        h_cnn = self.cnn_encoder(node_cblv)
        h_aux = self.aux_branch(node_aux)
        h = torch.cat([h_cnn, h_aux], dim=1)
        h = self.gat_layer(g, h).flatten(1)

        for layer in self.classifier[:-1]:
            h = self.act_fn(layer(h))
        return self.classifier[-1](h).squeeze(-1)
