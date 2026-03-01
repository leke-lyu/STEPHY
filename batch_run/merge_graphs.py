#!/usr/bin/env python3
"""Merge per-batch graphs.pt files into a single graphs.pt."""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description='Merge batch graphs.pt files')
    parser.add_argument('--result_dir', required=True,
                        help='Result directory containing batch_*_graphs.pt files')
    parser.add_argument('--output', default=None,
                        help='Output path (writes to result_dir/graphs.pt if omitted)')
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    batch_files = sorted(result_dir.glob('batch_*_graphs.pt'))

    if not batch_files:
        print(f"Error: No batch_*_graphs.pt found in {result_dir}")
        exit(1)

    print(f"Found {len(batch_files)} batch files:")
    all_graphs = []
    for f in batch_files:
        graphs = torch.load(f, weights_only=False)
        print(f"  {f}: {len(graphs)} graphs")
        all_graphs.extend(graphs)

    output_path = Path(args.output) if args.output else result_dir / 'graphs.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_graphs, output_path)
    print(f"Merged {len(all_graphs)} graphs -> {output_path}")


if __name__ == '__main__':
    main()
