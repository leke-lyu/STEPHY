#!/usr/bin/env python3
"""
Benchmark STEPHY single-tree inference latency (forward pass only).

For each label, the script:
  1. Loads the trained model from <model_dir>/<label>/best_model.pt
  2. Identifies the test split via <model_dir>/cls_as/test_predictions.csv
     (the 4 labels share a seed -> identical test set)
  3. Streams batch_*_graphs.pt files, normalises aux/edge features per the
     saved norm_params.pt, and times model.forward() one graph at a time.
  4. Discards the first n_warmup non-test graphs as warmup, so the
     reported n equals the full test-set size.

All timing is single-threaded (torch.set_num_threads(1)) with batch=1 to
report honest per-tree latency suitable for a Methods section.
"""

import argparse
import gc
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make stephy/ importable
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / 'stephy'))

from config import get_config              # noqa: E402
from model import CBLV_GAT                 # noqa: E402
from model_core import count_parameters    # noqa: E402


DEFAULT_GRAPHS_DIR = '/projects/lau_projects/simu/100k_diverse_population_result'
DEFAULT_MODEL_DIR = '/projects/lau_projects/simu/100k_diverse_population_result/stephy'
DEFAULT_LABELS = 'cls_as,reg_r0,reg_rr,reg_sss'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graphs_dir', default=DEFAULT_GRAPHS_DIR,
                   help='Directory containing batch_*_graphs.pt')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR,
                   help='STEPHY model dir; expects <label>/best_model.pt + norm_params.pt')
    p.add_argument('--labels', default=DEFAULT_LABELS,
                   help='Comma-separated list of labels to benchmark')
    p.add_argument('--n_warmup', type=int, default=20,
                   help='Warmup forward passes per label (discarded from stats)')
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
        import dgl
        print(f'  DGL             : {dgl.__version__}')
    except ImportError:
        pass
    print(f'  NumPy           : {np.__version__}')
    print(f'  torch.threads   : {torch.get_num_threads()}   (pinned for single-core timing)')
    print()


# ---------------- test set membership ----------------

def load_test_triplets(model_dir):
    """Read cls_as/test_predictions.csv -> set of (batch, sim_id, tree_idx)."""
    csv_path = Path(model_dir) / 'cls_as' / 'test_predictions.csv'
    df = pd.read_csv(csv_path, usecols=['batch', 'sim_id', 'tree_idx'])
    return {(str(b), str(s), int(t))
            for b, s, t in df.itertuples(index=False, name=None)}


def meta_key(meta):
    return (str(meta['batch']), str(meta['sim_id']), int(meta['tree_idx']))


# ---------------- model loading ----------------

def has_cp(label_dir):
    return (Path(label_dir) / 'cp_calibration.pt').exists()


def build_model(label, label_dir, subtree_width):
    config = get_config()
    model_args = dict(config['model'])
    model_args['subtree_width'] = subtree_width

    is_cls = label.startswith('cls_')
    if not is_cls and has_cp(label_dir):
        model_args['num_outputs'] = len(config['train']['cqr_quantiles'])
    else:
        model_args['num_outputs'] = 1

    model = CBLV_GAT(model_args)
    state = torch.load(str(Path(label_dir) / 'best_model.pt'),
                       map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, model_args['num_outputs']


# ---------------- graph collection ----------------

def collect_graphs(graphs_dir, test_triplets, n_warmup):
    """Stream batch files; return (test_records, warmup_graphs).

    test_records is a list of (g, n_tips_total, sim_id_str) tuples, where
    n_tips_total is the sum of per-location n_tips (5th aux feature) — the
    total number of sampled tips in the original BEAST2 tree. Aux features
    in g.ndata['aux'] are raw at this stage; normalisation is applied
    out-of-place during timing, so reading them here is safe.
    """
    graphs_dir = Path(graphs_dir)
    batch_files = sorted(graphs_dir.glob('batch_*_graphs.pt'))
    if not batch_files:
        raise FileNotFoundError(f'No batch_*_graphs.pt in {graphs_dir}')

    test_records, warmup_graphs = [], []
    print(f'Streaming {len(batch_files)} batch files ...', flush=True)
    for bf in batch_files:
        batch = torch.load(str(bf), weights_only=False)
        for g, meta, locs, height in batch:
            if meta_key(meta) in test_triplets:
                n_tips = int(g.ndata['aux'][:, 4].sum().item())
                sim_id = f"{meta['batch']}/{meta['sim_id']}_{meta['tree_idx']}"
                test_records.append((g, n_tips, sim_id))
            elif len(warmup_graphs) < n_warmup:
                warmup_graphs.append(g)
        del batch
        gc.collect()
    print(f'  collected: {len(test_records)} test, {len(warmup_graphs)} warmup',
          flush=True)
    return test_records, warmup_graphs


# ---------------- per-graph normalisation ----------------

def normed_features(g, aux_mean, aux_std, edge_mean, edge_std):
    """Return (cblv, aux, edge_feat) tensors with norms applied (out-of-place)."""
    cblv = g.ndata['cblv']

    aux = g.ndata['aux']
    aux = (torch.log(aux.clamp(min=1e-8)) - aux_mean) / aux_std

    feat = g.edata['feat'].clone()
    feat[:, 0] = torch.log(feat[:, 0].clamp(min=1e-8))
    feat[:, 2] = torch.log(feat[:, 2].clamp(min=1e-8))
    feat = (feat - edge_mean) / edge_std

    return cblv, aux, feat


# ---------------- per-label benchmark ----------------

def bench_label(label, model_dir, test_records, warmup_graphs):
    label_dir = Path(model_dir) / label
    norm_params = torch.load(str(label_dir / 'norm_params.pt'),
                             map_location='cpu', weights_only=False)
    aux_mean = norm_params['aux']['mean']
    aux_std = norm_params['aux']['std']
    edge_mean = norm_params['edge']['mean']
    edge_std = norm_params['edge']['std']

    subtree_width = test_records[0][0].ndata['cblv'].shape[2]
    model, out_dim = build_model(label, label_dir, subtree_width)
    n_params = count_parameters(model)

    latencies_ns = []
    print(f'  {label}: warmup ({len(warmup_graphs)}) + time ({len(test_records)}) ...',
          flush=True)

    with torch.no_grad():
        for g in warmup_graphs:
            cblv, aux, ef = normed_features(g, aux_mean, aux_std, edge_mean, edge_std)
            _ = model(g, cblv, aux, ef)

        for g, _n_tips, _sim_id in test_records:
            cblv, aux, ef = normed_features(g, aux_mean, aux_std, edge_mean, edge_std)
            t0 = time.perf_counter_ns()
            _ = model(g, cblv, aux, ef)
            t1 = time.perf_counter_ns()
            latencies_ns.append(t1 - t0)

    return {
        'label': label,
        'n_params': n_params,
        'out_dim': out_dim,
        'latencies_ms': np.array(latencies_ns) / 1e6,
        'n_tips': np.array([r[1] for r in test_records]),
        'sim_ids': [r[2] for r in test_records],
    }


# ---------------- reporting ----------------

def format_row(r):
    lat = r['latencies_ms']
    return (
        f"  {r['label']:<10}    "
        f"{r['n_params']/1e6:5.2f} M     "
        f"{r['out_dim']:>2}     "
        f"{lat.mean():7.2f}   "
        f"{lat.std():5.2f}  "
        f"{lat.min():5.2f}  "
        f"{np.median(lat):5.2f}  "
        f"{lat.max():6.2f}    "
        f"{lat.sum()/1e3:7.1f}    "
        f"{1e3/lat.mean():6.1f}"
    )


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    print('=' * 72)
    print('STEPHY inference benchmark')
    print('=' * 72)
    print()

    print_system_block()

    test_triplets = load_test_triplets(args.model_dir)
    print('Test set')
    print(f'  Source          : {args.graphs_dir}')
    print(f'  Membership      : {Path(args.model_dir) / "cls_as" / "test_predictions.csv"}')
    print(f'  Test graphs     : {len(test_triplets)}')
    n_batches = len(sorted(Path(args.graphs_dir).glob('batch_*_graphs.pt')))
    print(f'  Batches read    : {n_batches}  (streamed once into RAM)')
    print(f'  Warmup          : {args.n_warmup} non-test graphs per label (discarded)')
    print()

    test_records, warmup_graphs = collect_graphs(
        args.graphs_dir, test_triplets, args.n_warmup)
    if len(test_records) != len(test_triplets):
        print(f'  WARNING: collected {len(test_records)} of '
              f'{len(test_triplets)} expected test graphs')
    if len(warmup_graphs) < args.n_warmup:
        print(f'  WARNING: only {len(warmup_graphs)}/{args.n_warmup} warmup graphs found')

    labels = [s.strip() for s in args.labels.split(',') if s.strip()]
    results = [bench_label(label, args.model_dir, test_records, warmup_graphs)
               for label in labels]

    n_test = len(results[0]['latencies_ms'])
    print()
    print(f'Per-label timing  (forward pass only, batch=1, n={n_test} after {args.n_warmup} warmup)')
    print('  label         params    out_dim   mean ms     std    min    p50     max    total s    trees/s')
    for r in results:
        print(format_row(r))
    print()

    # Tree-size dependence: latency at the slowest and fastest tree, plus
    # Pearson correlation with n_tips. Tree size shouldn't affect latency
    # because the model input is fixed-shape (CBLV padded to subtree_width).
    n_tips_all = results[0]['n_tips']
    print(f'Tree-size dependence  (n_tips = total sampled tips per tree, range {n_tips_all.min()}-{n_tips_all.max()})')
    print('  label        slowest tree                       fastest tree                       Pearson r')
    print('               ms      n_tips   sim_id            ms      n_tips   sim_id            (latency vs n_tips)')
    for r in results:
        lat = r['latencies_ms']
        tips = r['n_tips']
        sids = r['sim_ids']
        i_max = int(np.argmax(lat))
        i_min = int(np.argmin(lat))
        corr = float(np.corrcoef(lat, tips)[0, 1])
        print(f"  {r['label']:<10}   "
              f"{lat[i_max]:6.2f}  {tips[i_max]:6d}   {sids[i_max]:<16}  "
              f"{lat[i_min]:6.2f}  {tips[i_min]:6d}   {sids[i_min]:<16}  "
              f"r = {corr:+.3f}")
    print()

    all_lat = np.concatenate([r['latencies_ms'] for r in results])
    cpu_name = detect_cpu_model()
    print('Paper summary')
    print(f'  On a single core of {cpu_name},')
    print(f'  STEPHY produces one estimate per tree in '
          f'{all_lat.mean():.2f} +/- {all_lat.std():.2f} ms')
    print(f'  (mean +/- std across {len(labels)} labels x {n_test} test trees, batch=1, fp32).')
    tput_str = ', '.join(f"{r['label']}={1e3/r['latencies_ms'].mean():.0f}/s"
                          for r in results)
    print(f'  Per-label throughput: {tput_str}.')
    print('=' * 72)


if __name__ == '__main__':
    main()
