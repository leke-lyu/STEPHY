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

    # Discard trees where any location has more tips than subtree_width
    filtered = []
    discarded = 0
    for g, graph_id, locs, height in graphs:
        max_tips = int(g.ndata['aux'][:, 4].max().item())
        if max_tips <= args.subtree_width:
            filtered.append((g, graph_id, locs, height))
        else:
            discarded += 1
            print(f"Discarded {graph_id}: max tips/location = {max_tips} > {args.subtree_width}")

    if discarded:
        print(f"Kept {len(filtered)}/{len(graphs)} graphs ({discarded} discarded)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
