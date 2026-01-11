#!/usr/bin/env python3
"""
Simplified CNN for phylogenetic parameter estimation.
No auxiliary data branch, point estimates only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedPhyloNet(nn.Module):
    """
    Phylogenetic CNN with three parallel branches:
    - Standard: local features
    - Stride: hierarchical patterns
    - Dilate: long-range dependencies
    """

    def __init__(self, input_channels, num_outputs, args):
        super(SimplifiedPhyloNet, self).__init__()

        self.input_channels = input_channels
        self.num_outputs = num_outputs

        # Branch configs
        self.phy_channel_plain = list(args['phy_channel_plain'])
        self.phy_channel_stride = list(args['phy_channel_stride'])
        self.phy_channel_dilate = list(args['phy_channel_dilate'])
        self.phy_kernel_plain = list(args['phy_kernel_plain'])
        self.phy_kernel_stride = list(args['phy_kernel_stride'])
        self.phy_kernel_dilate = list(args['phy_kernel_dilate'])
        self.phy_stride_stride = list(args['phy_stride_stride'])
        self.phy_dilate_dilate = list(args['phy_dilate_dilate'])
        self.lbl_channel = list(args['lbl_channel'])

        # Activation function
        act_name = args.get('activation_func', 'relu')
        self.act_fn = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'elu': F.elu}.get(act_name, F.relu)

        # Standard branch
        self.std_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel in zip(self.phy_channel_plain, self.phy_kernel_plain):
            self.std_convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding='same'))
            in_ch = out_ch
        self.std_pool = nn.AdaptiveAvgPool1d(1)

        # Stride branch
        self.stride_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, stride in zip(self.phy_channel_stride,
                                          self.phy_kernel_stride,
                                          self.phy_stride_stride):
            self.stride_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
            in_ch = out_ch
        self.stride_pool = nn.AdaptiveAvgPool1d(1)

        # Dilate branch
        self.dilate_convs = nn.ModuleList()
        in_ch = input_channels
        for out_ch, kernel, dilation in zip(self.phy_channel_dilate,
                                            self.phy_kernel_dilate,
                                            self.phy_dilate_dilate):
            self.dilate_convs.append(nn.Conv1d(in_ch, out_ch, kernel,
                                               dilation=dilation, padding='same'))
            in_ch = out_ch
        self.dilate_pool = nn.AdaptiveAvgPool1d(1)

        # Concatenation size
        self.concat_size = (self.phy_channel_plain[-1] +
                           self.phy_channel_stride[-1] +
                           self.phy_channel_dilate[-1])

        # Output head
        self.output_layers = nn.ModuleList()
        in_features = self.concat_size
        for out_features in self.lbl_channel:
            self.output_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.output_layers.append(nn.Linear(in_features, num_outputs))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: (batch, channels, tree_width) -> (batch, num_outputs)"""
        # Standard branch
        x_std = x
        for conv in self.std_convs:
            x_std = self.act_fn(conv(x_std))
        x_std = self.std_pool(x_std).squeeze(-1)

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

        # Concatenate and output
        x_concat = torch.cat([x_std, x_stride, x_dilate], dim=1)
        x_out = x_concat
        for layer in self.output_layers[:-1]:
            x_out = self.act_fn(layer(x_out))
        return self.output_layers[-1](x_out)


class PhyloDataset(torch.utils.data.Dataset):
    """Dataset for phylogenetic tensors."""

    def __init__(self, phy_data, labels):
        self.phy_data = torch.FloatTensor(phy_data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.phy_data[idx], self.labels[idx]
