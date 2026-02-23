#!/usr/bin/env python3
"""
STEPHY2 Model: CBLV-GAT with Phylogenetic Features + Aux Branch.

Node embedding: 96-dim CNN (48+24+24) + 32-dim aux branch = 128-dim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

_ACTIVATIONS = {'relu': F.relu}


def _kaiming_init(module):
    """Apply Kaiming initialization to Conv1d and Linear layers."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CBLVConvEncoder(nn.Module):
    """
    Convolutional encoder for CBLV subtree features.

    Three parallel branches:
    - Plain: Local features with standard convolutions
    - Stride: Hierarchical patterns with strided convolutions
    - Dilate: Long-range dependencies with dilated convolutions

    Input: (batch, 4, subtree_width) - CBLV features per node
    Output: (batch, 96) - Phylogenetic embedding (48+24+24)
    """

    def __init__(self, args):
        super().__init__()

        input_channels = 4  # CBLV has 4 channels

        # Branch configs from args
        phy_channel_plain = list(args['phy_channel_plain'])
        phy_channel_stride = list(args['phy_channel_stride'])
        phy_channel_dilate = list(args['phy_channel_dilate'])
        phy_kernel_plain = list(args['phy_kernel_plain'])
        phy_kernel_stride = list(args['phy_kernel_stride'])
        phy_kernel_dilate = list(args['phy_kernel_dilate'])
        phy_stride_stride = list(args['phy_stride_stride'])
        phy_dilate_dilate = list(args['phy_dilate_dilate'])

        # Activation function
        self.act_fn = _ACTIVATIONS.get(args['activation_func'], F.relu)

        # Plain branch: [12, 24, 48], kernels [3, 5, 7]
        self.plain_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel in zip(phy_channel_plain, phy_kernel_plain):
            self.plain_convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding='same'))
            in_ch = out_ch
        self.plain_pool = nn.AdaptiveAvgPool1d(1)

        # Stride branch: [12, 24], kernels [7, 9], strides [3, 6]
        self.stride_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, stride in zip(phy_channel_stride,
                                          phy_kernel_stride,
                                          phy_stride_stride):
            self.stride_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
            in_ch = out_ch
        self.stride_pool = nn.AdaptiveAvgPool1d(1)

        # Dilate branch: [12, 24], kernels [3, 5], dilations [3, 5]
        self.dilate_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, dilation in zip(phy_channel_dilate,
                                            phy_kernel_dilate,
                                            phy_dilate_dilate):
            self.dilate_convs.append(nn.Conv1d(in_ch, out_ch, kernel,
                                               dilation=dilation, padding='same'))
            in_ch = out_ch
        self.dilate_pool = nn.AdaptiveAvgPool1d(1)

        # Output dimension: 48 + 24 + 24 = 96
        self.output_dim = (phy_channel_plain[-1] +
                          phy_channel_stride[-1] +
                          phy_channel_dilate[-1])

        _kaiming_init(self)

    def _forward_branch(self, x, convs, pool):
        """Run a conv-pool branch: apply convolutions with activation, then pool."""
        for conv in convs:
            x = self.act_fn(conv(x))
        return pool(x).squeeze(-1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 4, subtree_width) CBLV features

        Returns:
            (batch, 96) node embeddings
        """
        x_plain = self._forward_branch(x, self.plain_convs, self.plain_pool)
        x_stride = self._forward_branch(x, self.stride_convs, self.stride_pool)
        x_dilate = self._forward_branch(x, self.dilate_convs, self.dilate_pool)

        # Concatenate: 48 + 24 + 24 = 96
        return torch.cat([x_plain, x_stride, x_dilate], dim=1)


class GraphEdgeAttention(nn.Module):
    """
    Graph attention layer with edge-based attention.

    Computes attention weights from edge features (DTW),
    aggregates neighbor embeddings, concatenates with self.

    Input: node embeddings (N, hidden_dim), edge features (E, 3)
    Output: (N, hidden_dim * 2)
    """

    def __init__(self, hidden_dim, edge_dim, attn_dim):
        super().__init__()

        # Edge attention network: edge_feat -> attention score
        self.attn_fc = nn.Sequential(
            nn.Linear(edge_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1)
        )

        _kaiming_init(self)

    def forward(self, g, h, edge_feat):
        """
        Forward pass.

        Args:
            g: DGL graph
            h: Node embeddings (N, hidden_dim)
            edge_feat: Edge features (E, edge_dim)

        Returns:
            (N, hidden_dim * 2) - concatenation of self and aggregated neighbor
        """
        with g.local_scope():
            g.ndata['h'] = h

            # Compute attention weights from edge features
            attn_scores = self.attn_fc(edge_feat)  # (E, 1)
            g.edata['a'] = dgl.ops.edge_softmax(g, attn_scores)

            # Message passing: weighted sum of neighbor embeddings
            g.update_all(
                fn.u_mul_e('h', 'a', 'm'),  # message = h * attention
                fn.sum('m', 'agg')           # aggregate = sum
            )

            # Concatenate self embedding with aggregated neighbors
            agg = g.ndata['agg']
            out = torch.cat([h, agg], dim=1)

            return out


class AuxBranch(nn.Module):
    """Dense branch for auxiliary tree statistics.

    Features: mrca_depth, earliest_tip_time, latest_tip_time, avg_branch_length, n_tips.

    Input: (batch, 5)
    Output: (batch, aux_output)
    """

    def __init__(self, args):
        super().__init__()
        aux_hidden = args['aux_hidden']
        aux_output = args['aux_output']
        self.fc = nn.Sequential(
            nn.Linear(5, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, aux_output),
            nn.ReLU(),
        )
        self.output_dim = aux_output
        _kaiming_init(self)

    def forward(self, x):
        """Forward pass: (batch, 5) auxiliary stats -> (batch, aux_output)."""
        return self.fc(x)


class CBLV_GAT(nn.Module):
    """
    CBLV-GAT: CNN encoder + aux branch + GAT for single-task prediction.

    Architecture:
    1. CNN encoder processes each node's CBLV independently -> 96-dim
    2. Aux branch processes tree statistics -> 32-dim
    3. Concat CNN + aux -> 128-dim node embedding
    4. GAT aggregates spatial information using DTW edges -> 256-dim
    5. Classifier predicts target label per node

    Args:
        args: Config dict with model parameters
    """

    def __init__(self, args):
        super().__init__()

        # CNN encoder for CBLV features
        self.cnn_encoder = CBLVConvEncoder(args)
        cnn_output_dim = self.cnn_encoder.output_dim  # 96

        # Aux branch for tree statistics
        self.aux_branch = AuxBranch(args)
        node_dim = cnn_output_dim + self.aux_branch.output_dim

        # Graph attention layer for spatial aggregation
        edge_dim = args['edge_dim']
        attn_dim = args['attn_dim']
        self.graph_attention = GraphEdgeAttention(node_dim, edge_dim, attn_dim)

        # GAT output: concat(self_128, neighbor_agg_128) = 256
        gat_output_dim = node_dim * 2  # 256
        lbl_channel = list(args['lbl_channel'])

        # Classifier: 256 -> 128 -> 64 -> 32 -> 1
        self.classifier = nn.ModuleList()
        in_features = gat_output_dim
        for out_features in lbl_channel:
            self.classifier.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.classifier.append(nn.Linear(in_features, 1))

        # Activation
        self.act_fn = _ACTIVATIONS.get(args['activation_func'], F.relu)

        _kaiming_init(self.classifier)

    def forward(self, g, node_cblv, node_aux, edge_feat):
        """
        Forward pass.

        Args:
            g: DGL graph with N nodes
            node_cblv: (N, 4, subtree_width) CBLV features per node
            node_aux: (N, 5) auxiliary tree statistics per node
            edge_feat: (E, 3) DTW edge features

        Returns:
            (N,) predictions per node
        """
        # CNN encode each node's CBLV
        h_cnn = self.cnn_encoder(node_cblv)  # (N, 96)
        h_aux = self.aux_branch(node_aux)    # (N, 32)
        h = torch.cat([h_cnn, h_aux], dim=1) # (N, 128)

        # Graph attention message passing
        h = self.graph_attention(g, h, edge_feat)  # (N, 256)

        # Classifier
        for layer in self.classifier[:-1]:
            h = self.act_fn(layer(h))
        out = self.classifier[-1](h).squeeze(-1)  # (N,)

        return out


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
