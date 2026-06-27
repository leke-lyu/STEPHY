#!/usr/bin/env python3
"""
Benchmark STEPHY single-tree inference latency on the 5 Denmark clade ML trees
(forward pass only).

Companion to bench_stephy_inference.py, but the "test set" is the five real
Nextstrain-clade ML trees from the Denmark bootstrap pipeline rather than the
100k simulation split. For each clade x task the script:

  1. Builds the per-clade DGL graph once from stephy_input/<clade>/ml.trees
     (CBLV + 5 aux per location, DTW edge features) at the training-time
     subtree_width (5744) -- identical to bootstrap_uncertainty step 07.
  2. Loads the trained pe_old/stephy2 head for the task (num_outputs=1).
  3. Normalises aux/edge features once, runs n_warmup warmups, then times
     model.forward() over n_repeat reps and records per-rep latency.

Tasks: r0, rr, sss (regression) and as (ancestor-location classification).

Each Denmark graph has exactly 5 nodes (one per Danish region) with CBLV padded
to subtree_width, so the forward pass sees a fixed-shape input regardless of how
many tips the clade has -- latency is expected to be ~flat across clades
(571-tip 21I vs 7177-tip 21K). The latency-vs-n_tips block makes that explicit.

All timing is single-threaded (torch.set_num_threads(1)) with batch=1 to report
honest per-tree latency suitable for a Methods section.

Usage:
    python3 denmark_stephy_inference.py
    python3 denmark_stephy_inference.py --input_dir /path/to/stephy_input \
        --model_dir /path/to/pe_old/stephy2
"""

import argparse
import gc
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# OpenMP collision shim (macOS); harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

# Make stephy/ importable (self-contained -- no dependence on the
# bootstrap_uncertainty scripts, which hardcode local Mac paths).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / 'stephy'))

from config import get_config              # noqa: E402
from model import CBLV_GAT                 # noqa: E402
from model_core import count_parameters    # noqa: E402
import data as stephy_data                 # noqa: E402
import dgl                                 # noqa: E402


DEFAULT_INPUT_DIR = '/projects/lau_projects/denmark/stephy_input'
DEFAULT_MODEL_DIR = '/projects/lau_projects/denmark/pe_old/stephy2'
DEFAULT_CLADES = '20I,21I,21J,21K,21L'
DEFAULT_TASKS = 'r0,rr,sss,as'

# Trained with subtree_width=5744. The CBLV encoder's AdaptiveAvgPool1d(1)
# averages over the FULL width including padding zeros, so a different width
# silently biases the output. Always use the training-time width (matches
# bootstrap_uncertainty step 07).
TRAINING_SUBTREE_WIDTH = 5744


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_dir', default=DEFAULT_INPUT_DIR,
                   help='Directory containing <clade>/ml.trees')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR,
                   help='pe_old/stephy2 dir; expects <task>/best_model.pt + norm_params.pt')
    p.add_argument('--clades', default=DEFAULT_CLADES,
                   help='Comma-separated clades to benchmark')
    p.add_argument('--tasks', default=DEFAULT_TASKS,
                   help='Comma-separated tasks to benchmark')
    p.add_argument('--n_warmup', type=int, default=20,
                   help='Warmup forward passes per clade x task (discarded)')
    p.add_argument('--n_repeat', type=int, default=100,
                   help='Timed forward passes per clade x task')
    p.add_argument('--subtree_width', type=int, default=TRAINING_SUBTREE_WIDTH,
                   help='CBLV width (must match training; default 5744)')
    return p.parse_args()


# ---------------- system introspection ----------------

def detect_cpu_model():
    if platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == 'Darwin':
        try:
            return subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                text=True).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return 'unknown'


def detect_ram_gb():
    if platform.system() == 'Linux':
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) / (1024 ** 2)
        except OSError:
            pass
    return None


def detect_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except ImportError:
        return None, os.cpu_count()


def print_system_block():
    print('System')
    print(f'  Platform        : {platform.system()} {platform.release()}')
    print(f'  CPU model       : {detect_cpu_model()}')
    phys, logi = detect_cores()
    if phys is not None:
        print(f'  Cores           : {phys} physical / {logi} logical')
    else:
        print(f'  Cores           : {logi} logical')
    ram = detect_ram_gb()
    if ram is not None:
        print(f'  RAM             : {ram:.0f} GB')
    print(f'  Python          : {platform.python_version()}')
    print(f'  PyTorch         : {torch.__version__}')
    try:
        import dgl as _dgl
        print(f'  DGL             : {_dgl.__version__}')
    except ImportError:
        pass
    print(f'  NumPy           : {np.__version__}')
    print(f'  torch.threads   : {torch.get_num_threads()}   (pinned for single-core timing)')
    print()


# ---------------- graph construction (mirrors bootstrap step 07) ----------------

def build_graph(tree_file, subtree_width):
    """Build the per-clade DGL graph (no labels). Mirrors 07_run_stephy.py.

    Returns (graph, n_tips) where n_tips is the total sampled tips in the tree
    (sum of the per-location 5th aux feature, read raw before normalisation).
    """
    phy = stephy_data.load_tree(str(tree_file), 0)
    tree_height = max(nd.root_distance for nd in phy.leaf_node_iter())
    encoder = stephy_data.VirtualSubtreeEncoder(phy, tree_height)
    locations = encoder.get_all_locations()
    n = len(locations)

    node_cblv = np.zeros((n, subtree_width, 4))
    node_aux = np.zeros((n, 5))
    for i, loc in enumerate(locations):
        cblv, aux = encoder.encode_cblv(loc, subtree_width=subtree_width,
                                        cblv_scale='tree_height')
        node_cblv[i] = cblv
        node_aux[i] = aux
    node_cblv = node_cblv.transpose(0, 2, 1)  # (n, 4, W) for Conv1d

    src, dst, edge_feats, _ = stephy_data.compute_dtw_edge_features(
        str(tree_file), 0)
    g = dgl.graph((src, dst), num_nodes=n)
    g.ndata['cblv'] = torch.tensor(node_cblv, dtype=torch.float32)
    g.ndata['aux'] = torch.tensor(node_aux, dtype=torch.float32)
    g.ndata['location'] = torch.tensor(locations, dtype=torch.long)
    g.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)

    n_tips = int(g.ndata['aux'][:, 4].sum().item())
    return g, n_tips


def normalize_graph(g, norm_params):
    """Out-of-place feature normalisation onto a graph copy: aux log+zscore,
    edge channels 0,2 log then zscore. Same scheme as bootstrap step 07."""
    aux = g.ndata['aux']
    g.ndata['aux'] = (
        torch.log(aux.clamp(min=1e-8)) - norm_params['aux']['mean']
    ) / norm_params['aux']['std']

    feat = g.edata['feat'].clone()
    feat[:, 0] = torch.log(feat[:, 0].clamp(min=1e-8))
    feat[:, 2] = torch.log(feat[:, 2].clamp(min=1e-8))
    g.edata['feat'] = (
        feat - norm_params['edge']['mean']
    ) / norm_params['edge']['std']


# ---------------- model loading ----------------

def build_model(task, model_dir, subtree_width):
    """Load the pe_old/stephy2 head for a task (num_outputs=1, no CQR/CP)."""
    task_dir = Path(model_dir) / task
    cfg = dict(get_config()['model'])
    cfg['subtree_width'] = subtree_width
    cfg['num_outputs'] = 1

    model = CBLV_GAT(cfg)
    state = torch.load(str(task_dir / 'best_model.pt'),
                       map_location='cpu', weights_only=False)
    if 'gat_layer.fc.weight' in state:
        # Trip-wire: a legacy GATConv checkpoint would load into the wrong layer.
        sys.exit(f"task {task} has legacy 'gat_layer.fc.weight' key -- that "
                 f"arch isn't supported by this benchmark.")
    model.load_state_dict(state)
    model.eval()

    norm = torch.load(str(task_dir / 'norm_params.pt'),
                      map_location='cpu', weights_only=False)
    return model, norm


# ---------------- per clade x task benchmark ----------------

def bench_clade_task(g_raw, model, norm, n_warmup, n_repeat):
    """Time forward() on one clade's graph for one task. Returns latencies (ms).

    The graph is normalised once onto a copy (normalisation is not part of the
    forward pass), then forward() is timed n_repeat times on that fixed input.
    """
    import copy
    g = copy.deepcopy(g_raw)
    normalize_graph(g, norm)

    latencies_ns = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(g, g.ndata['cblv'], g.ndata['aux'], g.edata['feat'])
        for _ in range(n_repeat):
            t0 = time.perf_counter_ns()
            _ = model(g, g.ndata['cblv'], g.ndata['aux'], g.edata['feat'])
            t1 = time.perf_counter_ns()
            latencies_ns.append(t1 - t0)
    return np.array(latencies_ns) / 1e6


# ---------------- reporting ----------------

def format_row(clade, n_tips, task, n_params, lat):
    return (
        f"  {clade:<5}   {n_tips:6d}   {task:<5}   "
        f"{n_params/1e6:5.2f} M   "
        f"{lat.mean():7.2f}   "
        f"{lat.std():5.2f}  "
        f"{lat.min():5.2f}  "
        f"{np.median(lat):5.2f}  "
        f"{lat.max():6.2f}    "
        f"{1e3/lat.mean():6.1f}"
    )


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    clades = [c.strip() for c in args.clades.split(',') if c.strip()]
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]

    print('=' * 78)
    print('STEPHY Denmark clade inference benchmark')
    print('=' * 78)
    print()

    print_system_block()

    print('Test set')
    print(f'  Source          : {args.input_dir}')
    print(f'  Model           : {args.model_dir}')
    print(f'  Clades          : {", ".join(clades)}')
    print(f'  Tasks           : {", ".join(tasks)}')
    print(f'  subtree_width   : {args.subtree_width}')
    print(f'  Warmup / repeat : {args.n_warmup} / {args.n_repeat} per clade x task')
    print()

    # --- Build each clade's ML graph once (forward-only timing; build untimed) ---
    print('Building clade graphs ...', flush=True)
    graphs = {}
    for clade in clades:
        tree_file = Path(args.input_dir) / clade / 'ml.trees'
        if not tree_file.is_file():
            sys.exit(f'Missing {tree_file}')
        g_raw, n_tips = build_graph(tree_file, args.subtree_width)
        graphs[clade] = (g_raw, n_tips)
        print(f'  {clade}: {n_tips} tips', flush=True)
        gc.collect()
    print()

    # --- Load each task's model once, then time over every clade ---
    results = {}   # (clade, task) -> latencies ms
    n_params_by_task = {}
    for task in tasks:
        model, norm = build_model(task, args.model_dir, args.subtree_width)
        n_params_by_task[task] = count_parameters(model)
        print(f'  {task}: model loaded, timing {len(clades)} clades ...', flush=True)
        for clade in clades:
            g_raw, _ = graphs[clade]
            results[(clade, task)] = bench_clade_task(
                g_raw, model, norm, args.n_warmup, args.n_repeat)
    print()

    # --- Per clade x task timing table ---
    print(f'Per clade x task timing  (forward pass only, batch=1, '
          f'n_repeat={args.n_repeat} after {args.n_warmup} warmup)')
    print('  clade   n_tips   task    params    mean ms     std    min    p50     max    trees/s')
    for clade in clades:
        n_tips = graphs[clade][1]
        for task in tasks:
            print(format_row(clade, n_tips, task,
                             n_params_by_task[task], results[(clade, task)]))
    print()

    # --- Tree-size dependence per task: mean latency vs n_tips across clades ---
    n_tips_arr = np.array([graphs[c][1] for c in clades])
    print(f'Tree-size dependence  (mean latency vs n_tips across {len(clades)} '
          f'clades, n_tips range {n_tips_arr.min()}-{n_tips_arr.max()})')
    print('  task    mean ms @ smallest    mean ms @ largest    Pearson r (mean lat vs n_tips)')
    for task in tasks:
        means = np.array([results[(c, task)].mean() for c in clades])
        i_small = int(np.argmin(n_tips_arr))
        i_large = int(np.argmax(n_tips_arr))
        if np.std(means) < 1e-9:
            corr = float('nan')
        else:
            corr = float(np.corrcoef(means, n_tips_arr)[0, 1])
        print(f'  {task:<5}   {means[i_small]:8.2f} (n={n_tips_arr[i_small]:>4})   '
              f'   {means[i_large]:8.2f} (n={n_tips_arr[i_large]:>4})   '
              f'   r = {corr:+.3f}')
    print()

    # --- Paper summary ---
    all_lat = np.concatenate([results[k] for k in results])
    cpu_name = detect_cpu_model()
    print('Paper summary')
    print(f'  On a single core of {cpu_name},')
    print(f'  STEPHY produces one estimate per Denmark clade tree in '
          f'{all_lat.mean():.2f} +/- {all_lat.std():.2f} ms')
    print(f'  (mean +/- std across {len(tasks)} tasks x {len(clades)} clades '
          f'x {args.n_repeat} reps, batch=1, fp32).')
    tput = ', '.join(
        f'{task}={1e3/np.concatenate([results[(c, task)] for c in clades]).mean():.0f}/s'
        for task in tasks)
    print(f'  Per-task throughput: {tput}.')
    print('=' * 78)


if __name__ == '__main__':
    main()
