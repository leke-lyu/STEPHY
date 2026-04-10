#!/usr/bin/env python3
"""
Shared neural network components for STEPHY pipelines.

CBLVConvEncoder: 3-branch CNN for CBLV features (parameterized by config).
AuxBranch: MLP for auxiliary tree statistics (parameterized by config).

Used by all three pipelines (stephy, CBLV-CNN, CBLV-GAT).
Architecture is data-independent — parameter count is fixed regardless of
subtree_width or num_locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    3-branch convolutional encoder for CBLV subtree features.

    Branches capture complementary patterns from the CBLV matrix:
    - Plain: standard convolutions for local feature extraction
    - Stride: strided convolutions for hierarchical / multi-scale patterns
    - Dilate: dilated convolutions for long-range dependencies

    Input: (batch, 4, subtree_width)
    Output: (batch, output_dim) where output_dim = sum of branch final channels
    """

    def __init__(self, args):
        super().__init__()

        input_channels = 4
        self.act_fn = _ACTIVATIONS.get(args['activation_func'], F.relu)

        # Plain branch
        self.plain_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel in zip(args['phy_channel_plain'], args['phy_kernel_plain']):
            self.plain_convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding='same'))
            in_ch = out_ch
        self.plain_pool = nn.AdaptiveAvgPool1d(1)

        # Stride branch
        self.stride_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, stride in zip(args['phy_channel_stride'],
                                          args['phy_kernel_stride'],
                                          args['phy_stride_stride']):
            self.stride_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
            in_ch = out_ch
        self.stride_pool = nn.AdaptiveAvgPool1d(1)

        # Dilate branch
        self.dilate_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, dilation in zip(args['phy_channel_dilate'],
                                            args['phy_kernel_dilate'],
                                            args['phy_dilate_dilate']):
            self.dilate_convs.append(nn.Conv1d(in_ch, out_ch, kernel,
                                               dilation=dilation, padding='same'))
            in_ch = out_ch
        self.dilate_pool = nn.AdaptiveAvgPool1d(1)

        self.output_dim = (args['phy_channel_plain'][-1] +
                          args['phy_channel_stride'][-1] +
                          args['phy_channel_dilate'][-1])

        _kaiming_init(self)

    def _forward_branch(self, x, convs, pool):
        for conv in convs:
            x = self.act_fn(conv(x))
        return pool(x).squeeze(-1)

    def forward(self, x):
        """(batch, 4, subtree_width) -> (batch, output_dim)"""
        return torch.cat([
            self._forward_branch(x, self.plain_convs, self.plain_pool),
            self._forward_branch(x, self.stride_convs, self.stride_pool),
            self._forward_branch(x, self.dilate_convs, self.dilate_pool),
        ], dim=1)


class AuxBranch(nn.Module):
    """MLP for auxiliary tree statistics (mrca_depth, tip times, avg_bl, n_tips).

    Input: (batch, 5)
    Output: (batch, aux_output)
    """

    def __init__(self, args):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, args['aux_hidden']),
            nn.ReLU(),
            nn.Linear(args['aux_hidden'], args['aux_output']),
            nn.ReLU(),
        )
        self.output_dim = args['aux_output']
        _kaiming_init(self)

    def forward(self, x):
        return self.fc(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
