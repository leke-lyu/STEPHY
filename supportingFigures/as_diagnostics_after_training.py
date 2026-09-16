#!/usr/bin/env python3
"""
AS diagnostics (after training) — does the trained ancestral-state classifier
beat naive structural oracles on the test split?

Predictors of each tree's true ancestral state (MRCA location), all scored by
top-1 accuracy against true_ancestor:

  spillover_loc     the location with Spillover_Loc == 1 in {batch}/{sim_id}_nf.csv
                    (the simulated seed; dropped when not unique)
  earliest_tip_loc  the location of the smallest-time sampled tip in
                    {batch}/{sim_id}_beast2.trees, tree tree_idx
  pred_ancestor     the model's argmax prediction

Ancestral_State from _nf.csv is also read as a sanity check: it must equal
true_ancestor, and any disagreement is printed as a warning.

Outputs next to this script:
  as_diagnostics_after_training.csv  one row per test tree, all predictors joined
  as_diagnostics_after_training.out  tee'd stdout (accuracy table + warnings)

Defaults resolve from STEPHY_MODELS, the root of the Zenodo model archive
(see ../zenodo/README.md), which must contain simulation_benchmark/:
  --as-predictions  $STEPHY_MODELS/simulation_benchmark/100k_diverse_population_result/stephy/cls_as/test_predictions.csv
  --nf-root         $STEPHY_MODELS/simulation_benchmark/100k_diverse_population
The earliest-tip oracle needs the *_beast2.trees files under --nf-root.

Usage:
    export STEPHY_MODELS=/path/to/trained_model
    python3 as_diagnostics_after_training.py
    python3 as_diagnostics_after_training.py \\
        --as-predictions /path/to/cls_as/test_predictions.csv \\
        --nf-root        /path/to/batches/
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from _paths import under

KEY = ['batch', 'sim_id', 'tree_idx']
PREDICTORS = ['spillover_loc', 'earliest_tip_loc', 'pred_ancestor']
# BEAST2 tip annotation: N[&type="I{<loc>}",samp="sample",time=<t>]
TIP_RE = re.compile(r'\[&type="I\{(\d+)\}",samp="sample",time=([\d.eE+-]+)\]')


class _Tee:
    """Mirror writes across streams (tees stdout to as_diagnostics_after_training.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# -- Per-outbreak oracle loaders ---------------------------------------------

def flagged_locations(nf_path):
    """
    Return (spillover_loc, ancestral_state) from one {sim_id}_nf.csv.

    Each is the unique Location whose flag column equals 1, or None when zero
    or several rows carry the flag.
    """
    df = pd.read_csv(nf_path, usecols=['Location', 'Spillover_Loc', 'Ancestral_State'])

    def unique_hit(col):
        hits = df.loc[df[col] == 1, 'Location']
        return int(hits.iloc[0]) if len(hits) == 1 else None

    return unique_hit('Spillover_Loc'), unique_hit('Ancestral_State')


def earliest_tip_locs_from_trees(trees_path):
    """
    Return the earliest sampled tip's location for every tree in a NEXUS file.

    The list is in file order, so element `tree_idx` belongs to that tree.
    Smaller `time` means closer to the outbreak origin.
    """
    locs = []
    with open(trees_path) as fh:
        for line in fh:
            if line.lstrip().startswith('tree '):
                tips = TIP_RE.findall(line)
                locs.append(int(min(tips, key=lambda t: float(t[1]))[0]) if tips else None)
    return locs


# -- Join and score ----------------------------------------------------------

def attach_oracles(pred_df, nf_root):
    """
    Add spillover_loc, earliest_tip_loc and nf_ancestor columns to pred_df.

    Each outbreak's _nf.csv and _beast2.trees are read once, however many of
    its trees are in the test split. Raises if either file is missing.
    """
    cache = {}
    spillover, earliest, nf_anc = [], [], []
    for batch, sim_id, tree_idx in pred_df[KEY].itertuples(index=False):
        if (batch, sim_id) not in cache:
            sim_dir = Path(nf_root) / str(batch)
            nf_path = sim_dir / f'{sim_id}_nf.csv'
            trees_path = sim_dir / f'{sim_id}_beast2.trees'
            for path in (nf_path, trees_path):
                if not path.exists():
                    raise FileNotFoundError(f'Missing file: {path}')
            cache[(batch, sim_id)] = (*flagged_locations(nf_path),
                                      earliest_tip_locs_from_trees(trees_path))
        sp, anc, tips = cache[(batch, sim_id)]
        spillover.append(sp)
        nf_anc.append(anc)
        earliest.append(tips[tree_idx] if tree_idx < len(tips) else None)

    out = pred_df.copy()
    out['spillover_loc'] = spillover
    out['earliest_tip_loc'] = earliest
    out['nf_ancestor'] = nf_anc
    return out


def top1_accuracy(df, predictor_col, target_col='true_ancestor'):
    """Return (n, accuracy) over rows where the predictor is defined."""
    sub = df[df[predictor_col].notna()]
    if sub.empty:
        return 0, float('nan')
    return len(sub), float((sub[predictor_col].astype(int) == sub[target_col].astype(int)).mean())


# -- Main --------------------------------------------------------------------

def main():
    """
    Score the spillover and earliest-tip oracles and the model's prediction
    against true_ancestor on the cls_as test split; write the joined CSV and
    the accuracy table next to this script.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--as-predictions', type=Path,
                   default=under('models', 'simulation_benchmark',
                                 '100k_diverse_population_result', 'stephy',
                                 'cls_as', 'test_predictions.csv'),
                   help='cls_as/test_predictions.csv from a trained pipeline')
    p.add_argument('--nf-root', type=Path,
                   default=under('models', 'simulation_benchmark',
                                 '100k_diverse_population'),
                   help='Root containing batch_*/{sim_id}_nf.csv and {sim_id}_beast2.trees')
    args = p.parse_args()

    stem = Path(__file__).resolve().parent / 'as_diagnostics_after_training'
    log = open(stem.with_suffix('.out'), 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    pred_df = pd.read_csv(args.as_predictions)
    missing = set(KEY + ['true_ancestor', 'pred_ancestor']) - set(pred_df.columns)
    if missing:
        raise ValueError(f'{args.as_predictions} lacks columns {sorted(missing)}')

    enriched = attach_oracles(pred_df, args.nf_root)

    mismatch = enriched['nf_ancestor'].notna() & (
        enriched['nf_ancestor'].astype('Int64') != enriched['true_ancestor'].astype('Int64'))
    if mismatch.any():
        bad = enriched.loc[mismatch, KEY + ['true_ancestor', 'nf_ancestor']]
        print(f'WARN: {len(bad)} rows where _nf.csv Ancestral_State '
              f'disagrees with true_ancestor. First 5:')
        print(bad.head().to_string(index=False))

    enriched.to_csv(stem.with_suffix('.csv'), index=False)

    print(f'{"Predictor":<20} {"N":>8} {"top-1":>8}')
    for col in PREDICTORS:
        n, acc = top1_accuracy(enriched, col)
        print(f'{col:<20} {n:>8} {acc:>7.1%}')
    for ext in ('.csv', '.out'):
        print(f'Saved: {stem.with_suffix(ext)}')
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == '__main__':
    main()
