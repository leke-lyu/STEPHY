#!/usr/bin/env python3
"""
STEPHY Model: CBLV-GAT with edge-attention using DTW features.

CNN (96) + Aux (32) = 128-dim node embedding -> edge-attention GAT -> 256-dim -> classifier.
"""

import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from model_core import _ACTIVATIONS, _kaiming_init, CBLVConvEncoder, AuxBranch, count_parameters


class GraphEdgeAttention(nn.Module):
    """
    Edge-attention GAT layer using DTW features.

    Computes attention weights from edge features,
    aggregates neighbor embeddings, concatenates with self.

    Input: node embeddings (N, hidden_dim), edge features (E, 3)
    Output: (N, hidden_dim * 2)
    """

    def __init__(self, hidden_dim, edge_dim, attn_dim):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(edge_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1)
        )
        _kaiming_init(self)

    def forward(self, g, h, edge_feat):
        with g.local_scope():
            g.ndata['h'] = h
            attn_scores = self.attn_fc(edge_feat)
            g.edata['a'] = dgl.ops.edge_softmax(g, attn_scores)
            g.update_all(
                fn.u_mul_e('h', 'a', 'm'),
                fn.sum('m', 'agg')
            )
            return torch.cat([h, g.ndata['agg']], dim=1)


class CBLV_GAT(nn.Module):
    """
    STEPHY main model: CNN encoder + aux branch + edge-attention GAT + classifier.

    Architecture:
        1. CBLVConvEncoder: CBLV (4, W) -> 96-dim
        2. AuxBranch: 5-dim stats -> 32-dim
        3. Concat -> 128-dim node embedding
        4. GraphEdgeAttention (DTW edges) -> 256-dim
        5. Classifier: 256 -> 128 -> 64 -> 32 -> 1
    """

    def __init__(self, args):
        super().__init__()
        self.cnn_encoder = CBLVConvEncoder(args)
        self.aux_branch = AuxBranch(args)
        node_dim = self.cnn_encoder.output_dim + self.aux_branch.output_dim

        self.graph_attention = GraphEdgeAttention(node_dim, args['edge_dim'], args['attn_dim'])

        gat_output_dim = node_dim * 2
        self.act_fn = _ACTIVATIONS.get(args['activation_func'], torch.relu)

        self.classifier = nn.ModuleList()
        in_features = gat_output_dim
        for out_features in args['lbl_channel']:
            self.classifier.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.classifier.append(nn.Linear(in_features, 1))
        _kaiming_init(self.classifier)

    def forward(self, g, node_cblv, node_aux, edge_feat):
        """(g, CBLV, aux, edge_feat) -> (N,) per-node predictions."""
        h_cnn = self.cnn_encoder(node_cblv)
        h_aux = self.aux_branch(node_aux)
        h = torch.cat([h_cnn, h_aux], dim=1)
        h = self.graph_attention(g, h, edge_feat)

        for layer in self.classifier[:-1]:
            h = self.act_fn(layer(h))
        return self.classifier[-1](h).squeeze(-1)
