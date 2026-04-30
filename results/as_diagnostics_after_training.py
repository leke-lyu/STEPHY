#!/usr/bin/env python3
"""
AS diagnostics (after training) — does the trained ancestral-state classifier
beat naive structural oracles on the GNN's test split?

Three predictors of the per-tree true ancestor location:

  A  spillover_loc       (sim metadata; the unique location with
                          Spillover_Loc == 1 in {batch}/{sim_id}_nf.csv)
  B  earliest_tip_loc    (tree-only oracle; the location of the
                          smallest-time sampled tip in
                          {batch}/{sim_id}_beast2.trees, indexed by tree_idx)
  C  pred_ancestor       (the GNN's argmax prediction)

Each predictor is scored against `true_ancestor` from the test split:
  • Top-1 accuracy

Outputs (next to this script, or under --output-dir):
  as_diagnostics_after_training.csv  — one row per (batch, sim_id, tree_idx)
                                       with all three predictors joined
  stdout                             — per-predictor top-1 accuracy table

Usage:
    python3 as_diagnostics_after_training.py \\
        --as-predictions /path/to/cls_as/test_predictions.csv \\
        --nf-root        /path/to/batches/
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


class _Tee:
    """Forward writes to multiple text streams (terminal + .out logfile)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

KEY = ['batch', 'sim_id', 'tree_idx']
TIP_RE = re.compile(
    r'\[&type="I\{(\d+)\}",samp="sample",time=([\d.eE+-]+)\]'
)


# ─── per-sim oracle loaders (cached by caller) ─────────────────────────────

def spillover_loc_from_nf(nf_path):
    """Unique location index where Spillover_Loc == 1 in {sim_id}_nf.csv.

    Returns None if zero or more-than-one such row exists (ambiguous sims
    are dropped from the Spillover_Loc accuracy).
    """
    df = pd.read_csv(nf_path, usecols=['Location', 'Spillover_Loc'])
    hits = df.loc[df['Spillover_Loc'] == 1, 'Location']
    if len(hits) != 1:
        return None
    return int(hits.iloc[0])


def ancestral_state_from_nf(nf_path):
    """Unique location index where Ancestral_State == 1 in {sim_id}_nf.csv.

    Returned only as a sanity check against `true_ancestor` in the test
    predictions — they should always match.
    """
    df = pd.read_csv(nf_path, usecols=['Location', 'Ancestral_State'])
    hits = df.loc[df['Ancestral_State'] == 1, 'Location']
    if len(hits) != 1:
        return None
    return int(hits.iloc[0])


def earliest_tip_locs_from_trees(trees_path):
    """For each tree in a BEAST2 NEXUS file, return the location of the
    smallest-time sampled tip. Output: list aligned to tree order
    (so element `tree_idx` is the predictor for that tree).

    BEAST2 tip annotations look like
        N[&type="I{<loc>}",samp="sample",time=<t>]
    where smaller `time` means closer to the outbreak origin.
    """
    locs = []
    with open(trees_path) as fh:
        for line in fh:
            if not line.lstrip().startswith('tree '):
                continue
            best_t, best_loc = None, None
            for loc_s, t_s in TIP_RE.findall(line):
                t = float(t_s)
                if best_t is None or t < best_t:
                    best_t, best_loc = t, int(loc_s)
            locs.append(best_loc)
    return locs


# ─── join oracles onto the test split ──────────────────────────────────────

def attach_oracles(pred_df, nf_root):
    """Add `spillover_loc`, `earliest_tip_loc`, and a sanity-check
    `nf_ancestor` column to `pred_df` by reading per-sim files once."""
    nf_root = Path(nf_root)
    nf_cache, tree_cache = {}, {}
    spillover, earliest, nf_anc = [], [], []

    for _, row in pred_df.iterrows():
        batch, sim_id, tree_idx = row['batch'], row['sim_id'], int(row['tree_idx'])
        sim_dir = nf_root / str(batch)

        if (batch, sim_id) not in nf_cache:
            nf_path = sim_dir / f'{sim_id}_nf.csv'
            if not nf_path.exists():
                raise FileNotFoundError(f"Missing nf file: {nf_path}")
            nf_cache[(batch, sim_id)] = (
                spillover_loc_from_nf(nf_path),
                ancestral_state_from_nf(nf_path),
            )
        sp, anc = nf_cache[(batch, sim_id)]
        spillover.append(sp)
        nf_anc.append(anc)

        if (batch, sim_id) not in tree_cache:
            t_path = sim_dir / f'{sim_id}_beast2.trees'
            if not t_path.exists():
                raise FileNotFoundError(f"Missing trees file: {t_path}")
            tree_cache[(batch, sim_id)] = earliest_tip_locs_from_trees(t_path)
        tlocs = tree_cache[(batch, sim_id)]
        earliest.append(tlocs[tree_idx] if tree_idx < len(tlocs) else None)

    out = pred_df.copy()
    out['spillover_loc']    = spillover
    out['earliest_tip_loc'] = earliest
    out['nf_ancestor']      = nf_anc
    return out


# ─── scoring ───────────────────────────────────────────────────────────────

def predictor_summary(df, predictor_col, target_col='true_ancestor'):
    """Top-1 accuracy for a single predictor against `true_ancestor`.

    Rows where `predictor_col` is NaN/None are dropped (the predictor was
    undefined for that sim — e.g., ambiguous spillover_loc).
    """
    sub = df[df[predictor_col].notna()]
    n = len(sub)
    if n == 0:
        return {'n': 0, 'top1': float('nan')}
    correct = (sub[predictor_col].astype(int) == sub[target_col].astype(int)).mean()
    return {'n': n, 'top1': float(correct)}


# ─── entrypoint ────────────────────────────────────────────────────────────

def main():
    """
    Compare structural oracles (Spillover_Loc, earliest_tip_loc) against the
    GNN's `pred_ancestor` on the cls_as test split.

    For every (batch, sim_id, tree_idx) in the test predictions:
      • Look up Spillover_Loc and (sanity) Ancestral_State from
        {batch}/{sim_id}_nf.csv
      • Parse {batch}/{sim_id}_beast2.trees and take the smallest-time
        sampled tip's location for the requested tree_idx
    Then score each predictor's Top-1 accuracy against `true_ancestor`.

    Output: as_diagnostics_after_training.csv alongside this script + a
    stdout per-predictor accuracy table. No figure is produced.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--as-predictions', type=Path, required=True,
                   help='cls_as/test_predictions.csv from a trained pipeline')
    p.add_argument('--nf-root', type=Path, required=True,
                   help='Root containing batch_*/{sim_id}_nf.csv and '
                        '{sim_id}_beast2.trees')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'as_diagnostics_after_training.out'
    log_f = open(log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, log_f)

    pred_df = pd.read_csv(args.as_predictions)
    for col in KEY + ['true_ancestor', 'pred_ancestor']:
        if col not in pred_df.columns:
            raise ValueError(f"cls_as predictions missing '{col}'")

    enriched = attach_oracles(pred_df, args.nf_root)

    # Sanity check: nf_ancestor should match true_ancestor everywhere.
    mismatch = enriched['nf_ancestor'].notna() & (
        enriched['nf_ancestor'].astype('Int64')
        != enriched['true_ancestor'].astype('Int64')
    )
    if mismatch.any():
        bad = enriched.loc[mismatch, KEY + ['true_ancestor', 'nf_ancestor']]
        print(f"WARN: {len(bad)} rows where _nf.csv Ancestral_State "
              f"disagrees with true_ancestor. First 5:")
        print(bad.head().to_string(index=False))

    csv_path = out_dir / 'as_diagnostics_after_training.csv'
    enriched.to_csv(csv_path, index=False)

    print(f"{'Predictor':<20} {'N':>8} {'top-1':>8}")
    for col in ['spillover_loc', 'earliest_tip_loc', 'pred_ancestor']:
        s = predictor_summary(enriched, col)
        print(f"{col:<20} {s['n']:>8} {s['top1']:>7.1%}")

    print(f"Saved: {csv_path}")
    print(f"Saved: {log_path}")
    sys.stdout = sys.__stdout__
    log_f.close()


if __name__ == '__main__':
    main()
