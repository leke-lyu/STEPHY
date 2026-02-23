#!/usr/bin/env python3
"""Merge per-batch graphs.pt files into a single graphs.pt."""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description='Merge batch graphs.pt files')
    parser.add_argument('--data_dir', required=True,
                        help='Parent directory containing batch_* folders')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <data_dir>/graphs.pt)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    batch_files = sorted(data_dir.glob('batch_*/graphs.pt'))

    if not batch_files:
        print(f"Error: No batch_*/graphs.pt found in {data_dir}")
        exit(1)

    print(f"Found {len(batch_files)} batch files:")
    all_graphs = []
    for f in batch_files:
        graphs = torch.load(f, weights_only=False)
        print(f"  {f}: {len(graphs)} graphs")
        all_graphs.extend(graphs)

    output_path = Path(args.output) if args.output else data_dir / 'graphs.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_graphs, output_path)
    print(f"Merged {len(all_graphs)} graphs -> {output_path}")


if __name__ == '__main__':
    main()
