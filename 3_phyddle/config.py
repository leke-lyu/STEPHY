#!/usr/bin/env python3
"""
Phyddle config for estimating location-specific R0 from phylogenetic trees.

Usage:
    phyddle -c config.py -s F     # Format only
    phyddle -c config.py -s T     # Train only
    phyddle -c config.py -s E     # Estimate only
    phyddle -c config.py -s FTE   # All steps
"""

args = {
    # Workspace
    'prefix'             : 'r0_est',
    'sim_prefix'         : 'sim',
    'sim_dir'            : './sim_data',
    'fmt_dir'            : './format_output',
    'trn_dir'            : './train_output',
    'est_dir'            : './estimate_output',

    # Analysis
    'use_parallel'       : 'T',
    'use_cuda'           : 'F',
    'num_proc'           : -1,
    'no_emp'             : 'T',

    # Format
    'encode_all_sim'     : 'T',
    'num_char'           : 16,
    'num_states'         : 2,
    'min_num_taxa'       : 1400,
    'max_num_taxa'       : 6019,
    'tree_width'         : 6019,
    'tree_encode'        : 'serial',
    'brlen_encode'       : 'height_brlen',
    'char_encode'        : 'integer',
    'char_format'        : 'nexus',
    'tensor_format'      : 'hdf5',

    # Parameters to estimate
    'param_est'          : {
        'log_R0_0'  : 'num',
        'log_R0_1'  : 'num',
        'log_R0_2'  : 'num',
        'log_R0_3'  : 'num',
        'log_R0_4'  : 'num',
        'log_R0_5'  : 'num',
        'log_R0_6'  : 'num',
        'log_R0_7'  : 'num',
        'log_R0_8'  : 'num',
        'log_R0_9'  : 'num',
        'log_R0_10' : 'num',
        'log_R0_11' : 'num',
        'log_R0_12' : 'num',
        'log_R0_13' : 'num',
        'log_R0_14' : 'num',
        'log_R0_15' : 'num',
    },
    'param_data'         : {},

    # Train
    'num_epochs'         : 100,
    'num_early_stop'     : 3,
    'trn_batch_size'     : 64,
    'prop_test'          : 0.2,
    'prop_val'           : 0.1,
    'prop_cal'           : 0.1,
    'cpi_coverage'       : 0.95,
    'cpi_asymmetric'     : 'T',
    'loss_numerical'     : 'mse',
    'optimizer'          : 'adam',
    'learning_rate'      : 0.001,
    'activation_func'    : 'relu',
    'log_offset'         : 1.0,

    # CNN architecture
    'phy_channel_plain'  : [32, 64, 128],
    'phy_channel_stride' : [32, 64],
    'phy_channel_dilate' : [32, 64],
    'phy_kernel_plain'   : [3, 5, 7],
    'phy_kernel_stride'  : [7, 9],
    'phy_kernel_dilate'  : [3, 5],
    'phy_stride_stride'  : [3, 6],
    'phy_dilate_dilate'  : [3, 5],
    'aux_channel'        : [128, 64, 32],
    'lbl_channel'        : [128, 64, 32],

    'verbose'            : 'T',
}
