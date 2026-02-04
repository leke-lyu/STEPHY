#!/usr/bin/env python3
"""
CBLV-CNN Baseline Model: CNN-only (No Graph Structure).

Uses phylogenetic (CBLV encoder) features only, without graph attention.
Baseline for evaluating whether graph structure helps prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBLVConvEncoder(nn.Module):
    """
    Convolutional encoder for CBLV subtree features.

    Three parallel branches:
    - Plain: Local features with standard convolutions
    - Stride: Hierarchical patterns with strided convolutions
    - Dilate: Long-range dependencies with dilated convolutions

    Input: (batch, 4, subtree_width) - CBLV features per node
    Output: (batch, 256) - Phylogenetic embedding
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
        act_name = args['activation_func']
        self.act_fn = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'elu': F.elu}.get(act_name, F.relu)

        # Plain branch: [32, 64, 128], kernels [3, 5, 7]
        self.plain_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel in zip(phy_channel_plain, phy_kernel_plain):
            self.plain_convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding='same'))
            in_ch = out_ch
        self.plain_pool = nn.AdaptiveAvgPool1d(1)

        # Stride branch: [32, 64], kernels [7, 9], strides [3, 6]
        self.stride_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, stride in zip(phy_channel_stride,
                                          phy_kernel_stride,
                                          phy_stride_stride):
            self.stride_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
            in_ch = out_ch
        self.stride_pool = nn.AdaptiveAvgPool1d(1)

        # Dilate branch: [32, 64], kernels [3, 5], dilations [3, 5]
        self.dilate_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, dilation in zip(phy_channel_dilate,
                                            phy_kernel_dilate,
                                            phy_dilate_dilate):
            self.dilate_convs.append(nn.Conv1d(in_ch, out_ch, kernel,
                                               dilation=dilation, padding='same'))
            in_ch = out_ch
        self.dilate_pool = nn.AdaptiveAvgPool1d(1)

        # Output dimension: 128 + 64 + 64 = 256
        self.output_dim = (phy_channel_plain[-1] +
                          phy_channel_stride[-1] +
                          phy_channel_dilate[-1])

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
            (batch, 256) node embeddings
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

        # Concatenate: 128 + 64 + 64 = 256
        return torch.cat([x_plain, x_stride, x_dilate], dim=1)


class CBLV_CNN(nn.Module):
    """
    CBLV-CNN: CNN encoder + MLP for single-task prediction (no graph structure).

    Architecture:
    1. CNN encoder processes each node's CBLV independently -> 256-dim
    2. Classifier predicts target label per node

    Baseline for evaluating whether graph structure (GAT) helps prediction.

    Args:
        args: Config dict with model parameters
    """

    def __init__(self, args):
        super().__init__()

        # CNN encoder for CBLV features
        self.cnn_encoder = CBLVConvEncoder(args)
        cnn_output_dim = self.cnn_encoder.output_dim  # 256

        lbl_channel = list(args['lbl_channel'])

        # Classifier: 256 -> 128 -> 64 -> 32 -> 1
        self.classifier = nn.ModuleList()
        in_features = cnn_output_dim
        for out_features in lbl_channel:
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

    def forward(self, node_cblv):
        """
        Forward pass.

        Args:
            node_cblv: (N, 4, subtree_width) CBLV features per node

        Returns:
            (N,) predictions per node
        """
        # CNN encode each node's CBLV
        h = self.cnn_encoder(node_cblv)  # (N, 256)

        # Classifier
        for layer in self.classifier[:-1]:
            h = self.act_fn(layer(h))
        out = self.classifier[-1](h).squeeze(-1)  # (N,)

        return out


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
