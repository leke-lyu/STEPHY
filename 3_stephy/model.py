#!/usr/bin/env python3
"""
STEPHY Model: CBLV-GAT with Phylogenetic Features Only.

Uses phylogenetic (CBLV encoder) features with GAT for spatial aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class CBLVConvEncoder(nn.Module):
    """
    Convolutional encoder for CBLV subtree features.

    Three parallel branches:
    - Plain: Local features with standard convolutions
    - Stride: Hierarchical patterns with strided convolutions
    - Dilate: Long-range dependencies with dilated convolutions

    Input: (batch, 4, subtree_width) - CBLV features per node
    Output: (batch, 128) - Phylogenetic embedding
    """

    def __init__(self, args):
        super(CBLVConvEncoder, self).__init__()

        input_channels = 4  # CBLV has 4 channels

        # Branch configs from args
        self.phy_channel_plain = list(args['phy_channel_plain'])
        self.phy_channel_stride = list(args['phy_channel_stride'])
        self.phy_channel_dilate = list(args['phy_channel_dilate'])
        self.phy_kernel_plain = list(args['phy_kernel_plain'])
        self.phy_kernel_stride = list(args['phy_kernel_stride'])
        self.phy_kernel_dilate = list(args['phy_kernel_dilate'])
        self.phy_stride_stride = list(args['phy_stride_stride'])
        self.phy_dilate_dilate = list(args['phy_dilate_dilate'])

        # Activation function
        act_name = args['activation_func']
        self.act_fn = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'elu': F.elu}.get(act_name, F.relu)

        # Plain branch: [16, 32, 64], kernels [3, 5, 7]
        self.plain_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel in zip(self.phy_channel_plain, self.phy_kernel_plain):
            self.plain_convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding='same'))
            in_ch = out_ch
        self.plain_pool = nn.AdaptiveAvgPool1d(1)

        # Stride branch: [16, 32], kernels [7, 9], strides [3, 6]
        self.stride_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, stride in zip(self.phy_channel_stride,
                                          self.phy_kernel_stride,
                                          self.phy_stride_stride):
            self.stride_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
            in_ch = out_ch
        self.stride_pool = nn.AdaptiveAvgPool1d(1)

        # Dilate branch: [16, 32], kernels [3, 5], dilations [3, 5]
        self.dilate_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, dilation in zip(self.phy_channel_dilate,
                                            self.phy_kernel_dilate,
                                            self.phy_dilate_dilate):
            self.dilate_convs.append(nn.Conv1d(in_ch, out_ch, kernel,
                                               dilation=dilation, padding='same'))
            in_ch = out_ch
        self.dilate_pool = nn.AdaptiveAvgPool1d(1)

        # Output dimension: 64 + 32 + 32 = 128
        self.output_dim = (self.phy_channel_plain[-1] +
                          self.phy_channel_stride[-1] +
                          self.phy_channel_dilate[-1])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 4, subtree_width) CBLV features

        Returns:
            (batch, 128) node embeddings
        """
        # Plain branch
        x_plain = x
        for conv in self.plain_convs:
            x_plain = self.act_fn(conv(x_plain))
        x_plain = self.plain_pool(x_plain).squeeze(-1)

        # Stride branch
        x_stride = x
        for conv in self.stride_convs:
            x_stride = self.act_fn(conv(x_stride))
        x_stride = self.stride_pool(x_stride).squeeze(-1)

        # Dilate branch
        x_dilate = x
        for conv in self.dilate_convs:
            x_dilate = self.act_fn(conv(x_dilate))
        x_dilate = self.dilate_pool(x_dilate).squeeze(-1)

        # Concatenate: 64 + 32 + 32 = 128
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
        super(GraphEdgeAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Edge attention network: edge_feat -> attention score
        self.attn_fc = nn.Sequential(
            nn.Linear(edge_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


class CBLV_GAT(nn.Module):
    """
    CBLV-GAT: CNN encoder + GAT for single-task prediction.

    Architecture:
    1. CNN encoder processes each node's CBLV independently -> 128-dim
    2. GAT aggregates spatial information using DTW edges -> 256-dim
    3. Classifier predicts target label per node

    Args:
        args: Config dict with model parameters
    """

    def __init__(self, args):
        super(CBLV_GAT, self).__init__()

        self.subtree_width = args['subtree_width']

        # CNN encoder for CBLV features
        self.cnn_encoder = CBLVConvEncoder(args)
        cnn_output_dim = self.cnn_encoder.output_dim  # 128

        # Graph attention layer for spatial aggregation
        edge_dim = args['edge_dim']
        attn_dim = args['attn_dim']
        self.graph_attention = GraphEdgeAttention(cnn_output_dim, edge_dim, attn_dim)

        # GAT output: concat(self_128, neighbor_agg_128) = 256
        gat_output_dim = cnn_output_dim * 2  # 256
        self.lbl_channel = list(args['lbl_channel'])

        # Classifier: 256 -> 128 -> 64 -> 32 -> 1
        self.classifier = nn.ModuleList()
        in_features = gat_output_dim
        for out_features in self.lbl_channel:
            self.classifier.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.classifier.append(nn.Linear(in_features, 1))

        # Activation
        act_name = args['activation_func']
        self.act_fn = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'elu': F.elu}.get(act_name, F.relu)

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, g, node_cblv, edge_feat):
        """
        Forward pass.

        Args:
            g: DGL graph with N nodes
            node_cblv: (N, 4, subtree_width) CBLV features per node
            edge_feat: (E, 3) DTW edge features

        Returns:
            (N,) predictions per node
        """
        # CNN encode each node's CBLV
        h = self.cnn_encoder(node_cblv)  # (N, 128)

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
