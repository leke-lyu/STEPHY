#!/usr/bin/env python3
"""Build and save DGL graphs for STEPHY.

Wraps data.build_all_graphs() with CLI argument parsing and post-build
filtering.  Designed to be called per-batch by submit_build_graphs.sh.
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'stephy'))
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

    # Discard trees where any location has more tips than subtree_width.
    # Column 4 of the aux node features is n_tips (per-location tip count).
    # Trees exceeding subtree_width were zero-padded during CBLV encoding,
    # so their representations are lossy; filtering them avoids noisy inputs.
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
