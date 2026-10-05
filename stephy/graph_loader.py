"""Utility for loading graphs from a single .pt file or a directory of batch files."""

import gc
from pathlib import Path

import torch


def load_graphs(path):
    """Load graphs from a .pt file or a directory of batch_*_graphs.pt files.

    Parameters
    ----------
    path : str or Path
        Either a single ``graphs.pt`` file or a directory containing
        ``batch_*_graphs.pt`` files produced by ``submit_build_graphs.sh``.

    Returns
    -------
    list
        List of (graph, meta, locs, height) tuples, where ``meta`` is a dict
        with keys ``batch``, ``sim_id``, ``tree_idx``.
    """
    path = Path(path)

    if path.is_file():
        return torch.load(str(path), weights_only=False)

    if path.is_dir():
        batch_files = sorted(path.glob('batch_*_graphs.pt'))
        if not batch_files:
            raise FileNotFoundError(
                f"No batch_*_graphs.pt files found in {path}"
            )

        all_graphs = []
        print(f"Loading {len(batch_files)} batch files from {path}")
        for f in batch_files:
            graphs = torch.load(str(f), weights_only=False)
            print(f"  {f.name}: {len(graphs)} graphs")
            all_graphs.extend(graphs)
            del graphs
            gc.collect()

        print(f"Loaded {len(all_graphs)} graphs total")
        return all_graphs

    raise FileNotFoundError(f"Path does not exist: {path}")
