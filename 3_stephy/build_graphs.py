#!/usr/bin/env python3
"""Build and save DGL graphs for STEPHY."""

import argparse
from pathlib import Path
import torch
from data import build_all_graphs


def main():
    parser = argparse.ArgumentParser(description='STEPHY: Build graphs')
    parser.add_argument('--input_dir', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output file (e.g., graphs.pt)')
    parser.add_argument('--subtree_width', type=int, required=True, help='Max tips per location')
    args = parser.parse_args()

    graphs = build_all_graphs(args.input_dir, args.subtree_width, verbose=True)
    print(f"Graphs: {len(graphs)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
