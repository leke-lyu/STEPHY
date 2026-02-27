#!/usr/bin/env python3
"""Build and save DGL graphs for STEPHY."""

import argparse
from pathlib import Path
import torch
from config import DATA_ARGS
from data import build_all_graphs


def main():
    parser = argparse.ArgumentParser(description='STEPHY: Build graphs')
    parser.add_argument('--input_dir', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output file (e.g., graphs.pt)')
    parser.add_argument('--subtree_width', type=int, required=True, help='Max tips per location')
    parser.add_argument('--cblv_scale', default=DATA_ARGS['cblv_scale'],
                        choices=['tree_height', 'log1p'],
                        help='CBLV scaling method (default: %(default)s)')
    args = parser.parse_args()

    graphs = build_all_graphs(args.input_dir, args.subtree_width, verbose=True,
                              cblv_scale=args.cblv_scale)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
