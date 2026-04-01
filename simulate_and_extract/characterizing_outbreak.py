#!/usr/bin/env python3
"""
Outbreak Characterization Script.

Extracts per-location node features, ancestral-state label, and spillover
location from BEAST2 simulation outputs, writing a single CSV per simulation.

Processing order for each simulation:
  1. Read the phylogenetic tree (*_beast2.trees).
  2. Count sampled tips per location — skip the simulation if any location
     has 30 or fewer tips.
  3. Identify the ancestral location (MRCA of all sampled tips).
  4. Extract epidemic features and spillover location from the trajectory,
     reaction XML, and parameter files.
  5. Write a merged *_nf.csv.

Tip-count filtering uses regex on the Newick string (no tree parsing).
MRCA identification uses treeswift for fast tree traversal.
Event classification uses Numba-accelerated array diffing.

Input:  *_beast2.trees, *_beast2.traj, *_beast2.xml, *_parameter.csv
Output: *_nf.csv  (Location, Initial_Population, Epidemic_Peak, Peak_Timing,
                   Accumulated_Infections, R0, Recovery_Rate, Source_Sink_Score,
                   Ancestral_State, Spillover_Loc)

Usage:  python3 characterizing_outbreak.py /path/to/data [--summary-file FILE]
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import treeswift
from numba import jit
from tqdm import tqdm

# Matches I[n] in reaction strings, e.g. "I[0] + S[1] -> I[0] + I[1]"
I_PATTERN = re.compile(r'I\[(\d+)\]')

# Matches tip (leaf) location labels in the label-embedded Newick format.
# Tips are preceded by '(' or ','  while internal nodes are preceded by ')'.
TIP_LOC_PATTERN = re.compile(r'(?<=[(,])\s*\S+?__LOC(\d+)')


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _find_nonzero_changes(diff_arr):
    """Return indices and values of non-zero elements in a 2-D diff array."""
    total = 0
    for i in range(diff_arr.shape[0]):
        for j in range(diff_arr.shape[1]):
            if diff_arr[i, j] != 0:
                total += 1

    row_indices = np.empty(total, dtype=np.int32)
    col_indices = np.empty(total, dtype=np.int32)
    values = np.empty(total, dtype=np.int32)

    idx = 0
    for i in range(diff_arr.shape[0]):
        for j in range(diff_arr.shape[1]):
            if diff_arr[i, j] != 0:
                row_indices[idx] = i
                col_indices[idx] = j
                values[idx] = diff_arr[i, j]
                idx += 1

    return row_indices, col_indices, values


# ---------------------------------------------------------------------------
# Tree: reading, filtering, and label extraction
# ---------------------------------------------------------------------------

def _convert_annotation(match):
    """Convert a BEAST2 annotation ``[&I{n} time=t ...]`` to ``__LOCn``.

    This embeds the location directly into the node label so that treeswift
    (which discards ``[&...]`` comment blocks) can still access it.
    """
    loc = re.search(r'I\{(\d+)\}', match.group(0)).group(1)
    return f'__LOC{loc}'


def read_newick(filepath):
    """Read a BEAST2 .trees file and return the Newick string with
    annotations converted to the ``__LOCn`` label-embedded format.
    """
    content = filepath.read_text()
    raw = re.search(r'tree\s+\w+\s*=\s*(.+)', content).group(1)
    return re.sub(r'\[&[^\]]+\]', _convert_annotation, raw)


def check_tip_counts(newick, min_tips=30):
    """Return True if every location has more than *min_tips* sampled tips.

    Operates on the label-embedded Newick via regex -- no tree parsing needed.
    """
    counts = Counter(int(m.group(1)) for m in TIP_LOC_PATTERN.finditer(newick))
    return bool(counts) and all(c > min_tips for c in counts.values())


def extract_ancestral_labels(newick):
    """Identify the ancestral location and return per-location binary labels.

    Parses the Newick string with treeswift, finds the MRCA of all sampled
    tips (accounting for single-child migration nodes on the stem), and
    returns ``{location: 1}`` for the MRCA location and ``0`` for all others.

    Returns:
        dict[int, int]: Mapping of location ID to Ancestral_State (0 or 1).
    """
    tree = treeswift.read_tree_newick(newick)

    # Collect all locations present in the tree (internal + leaf nodes).
    locations = set()
    for node in tree.traverse_preorder():
        if node.label:
            m = re.search(r'__LOC(\d+)', node.label)
            if m:
                locations.add(int(m.group(1)))

    # MRCA of all sampled tips gives the ancestral (spillover) location.
    leaf_labels = {leaf.label for leaf in tree.traverse_leaves()}
    mrca = tree.mrca(leaf_labels)
    mrca_loc = int(re.search(r'__LOC(\d+)', mrca.label).group(1))

    return {loc: int(loc == mrca_loc) for loc in locations}


# ---------------------------------------------------------------------------
# Epidemic feature extraction
# ---------------------------------------------------------------------------

def _parse_reaction_side(side):
    """Parse one side of a reaction string (e.g. ``"I[0] + S[1]"``) into
    a dict of species -> coefficient.
    """
    species = defaultdict(int)
    for term in side.strip().split('+'):
        match = re.match(r'^(\d+)?(.+)$', term.strip())
        if match:
            coef = int(match.group(1)) if match.group(1) else 1
            species[match.group(2)] += coef
    return species


def load_trajectory(traj_file):
    """Load a BEAST2 trajectory file and pivot to wide format.

    Returns a pandas DataFrame with columns ``Sample``, ``t``, and one column
    per species (e.g. ``I[0]``, ``S[1]``, ``R``).

    Implementation note -- polars/pandas hybrid:
        Polars is used for the initial read and pivot because its eager
        ``pivot()`` is significantly faster than pandas ``pivot_table()``
        on the large, long-format trajectory files.  The result is
        converted to pandas (``.to_pandas()``) at the end because all
        downstream feature-extraction code (classify_events, epidemic
        peaks, etc.) relies on pandas indexing (e.g. ``.idxmax()``,
        ``.diff()``).
    """
    # -- Polars: fast read + pivot --
    df = pl.read_csv(traj_file, separator='\t')
    if df.is_empty():
        raise ValueError(f"Trajectory file '{traj_file}' is empty")

    # Build a composite species label, e.g. "I[0]", "S[1]", while
    # leaving scalar compartments ("R", "sample") as-is.
    df = df.with_columns(
        pl.when(pl.col('population').is_in(['R', 'sample']))
        .then(pl.col('population'))
        .otherwise(
            pl.col('population') + '['
            + pl.col('index').cast(pl.Int32).cast(pl.Utf8) + ']'
        )
        .alias('species')
    )
    # -- Convert to pandas for downstream compatibility --
    return (
        df.pivot(on='species', index=['Sample', 't'], values='value')
        .fill_null(0)
        .to_pandas()
    )


def load_parameters(param_file):
    """Load per-location R0, recovery rate, initial population, and spillover
    location from a parameter CSV."""
    df = pd.read_csv(param_file)
    r0 = {
        int(c.replace('R0_loc_', '')): float(df[c].iloc[0])
        for c in df.columns if c.startswith('R0_loc_')
    }
    recovery = {
        int(c.replace('recovery_rate_loc_', '')): float(df[c].iloc[0])
        for c in df.columns if c.startswith('recovery_rate_loc_')
    }
    pop = {
        int(c.replace('population_loc_', '')): int(df[c].iloc[0])
        for c in df.columns if c.startswith('population_loc_')
    }
    spillover = {
        int(c.replace('initial_infected_individuals_loc_', '')): int(df[c].iloc[0])
        for c in df.columns if c.startswith('initial_infected_individuals_loc_')
    }
    return {'R0': r0, 'Recovery_Rate': recovery, 'Initial_Population': pop, 'Spillover_Loc': spillover}


def load_reactions(xml_file):
    """Parse reaction definitions from a BEAST2 XML file.

    Returns a lookup table mapping net stoichiometric change (as a sorted
    tuple of (species, delta) pairs) to the original reaction string.
    """
    root = ET.parse(xml_file).getroot()
    lookup = {}

    for reaction in root.findall('.//reaction'):
        reaction_str = (reaction.text or '').strip()
        if '->' not in reaction_str:
            continue
        parts = reaction_str.split('->')
        if len(parts) != 2:
            continue

        reactants = _parse_reaction_side(parts[0])
        products = _parse_reaction_side(parts[1])
        all_species = set(reactants) | set(products)
        net_change = {s: products.get(s, 0) - reactants.get(s, 0) for s in all_species}
        lookup[tuple(sorted(net_change.items()))] = reaction_str

    return lookup


def calculate_epidemic_peaks(sample_data, num_nodes):
    """Return peak infected count and its timing for each location."""
    metrics = {n: {'peak': 0, 'peak_time': 0.0} for n in range(num_nodes)}
    i_cols = {c for c in sample_data.columns if c.startswith('I[')}

    for node_id in range(num_nodes):
        col = f'I[{node_id}]'
        if col in i_cols:
            peak = int(sample_data[col].max())
            if peak > 0:
                idx = sample_data[col].idxmax()
                metrics[node_id] = {
                    'peak': peak,
                    'peak_time': float(sample_data.loc[idx, 't']),
                }
    return metrics


def classify_events(sample_data, reaction_lookup, species_cols):
    """Match consecutive population changes to known reactions.

    Uses Numba-accelerated diffing to find non-zero changes between time
    steps, then looks up each change pattern in the reaction table.

    Returns a pandas Series of event-type counts.
    """
    diff_arr = sample_data[species_cols].diff().values[1:].astype(np.int32)
    col_names = list(species_cols)

    row_indices, col_indices, values = _find_nonzero_changes(diff_arr)
    if len(row_indices) == 0:
        return pd.Series(dtype=int)

    event_types = []
    current_row = -1
    current_changes = []

    for i in range(len(row_indices)):
        if row_indices[i] != current_row:
            if current_changes:
                event_types.append(
                    reaction_lookup.get(tuple(sorted(current_changes)), "Unknown")
                )
            current_row = row_indices[i]
            current_changes = []
        current_changes.append((col_names[col_indices[i]], int(values[i])))

    if current_changes:
        event_types.append(
            reaction_lookup.get(tuple(sorted(current_changes)), "Unknown")
        )

    return pd.Series(event_types).value_counts() if event_types else pd.Series(dtype=int)


def calculate_node_metrics(event_counts, num_nodes):
    """Derive accumulated infections and source/sink score per location.

    Source_Sink_Score = (exports - imports) / (exports + imports),
    ranging from -1 (pure sink) to +1 (pure source).
    """
    infections = np.zeros(num_nodes, dtype=int)
    exports = np.zeros(num_nodes, dtype=int)
    imports = np.zeros(num_nodes, dtype=int)

    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        left, right = [s.strip() for s in event_type.split('->', 1)]
        left_nodes = [int(n) for n in I_PATTERN.findall(left)]
        right_nodes = [int(n) for n in I_PATTERN.findall(right)]

        # Count new infections at each destination location.
        for node_id in right_nodes:
            if node_id < num_nodes:
                infections[node_id] += freq

        # Migration events: single I[src] -> I[dst] with src != dst.
        if (len(left_nodes) == 1 and len(right_nodes) == 1
                and left == f"I[{left_nodes[0]}]"
                and right == f"I[{right_nodes[0]}]"):
            src, dst = left_nodes[0], right_nodes[0]
            if src != dst:
                if src < num_nodes:
                    exports[src] += freq
                if dst < num_nodes:
                    imports[dst] += freq

    total = exports + imports
    source_sink = np.divide(
        exports - imports, total,
        out=np.zeros(num_nodes, dtype=float), where=total > 0,
    )

    return {
        n: {
            'accumulated_infections': int(infections[n]),
            'source_sink_score': float(source_sink[n]),
        }
        for n in range(num_nodes)
    }


# ---------------------------------------------------------------------------
# Simulation processing
# ---------------------------------------------------------------------------

def process_simulation(tree_file, traj_file, param_file, output_file, min_tips=30):
    """Process one simulation end-to-end.

    Reads the tree, checks the tip-count filter, extracts epidemic features
    from the trajectory and parameter files, and writes a merged CSV.

    Returns:
        One of: 'processed', 'skipped', 'error'.
    """
    try:
        # --- Tree: filter by tip count, then extract ancestral label ---
        newick = read_newick(tree_file)
        if not check_tip_counts(newick, min_tips):
            return 'skipped'
        ancestral_labels = extract_ancestral_labels(newick)

        # --- Epidemic features from trajectory / parameters / reactions ---
        params = load_parameters(str(param_file))
        num_nodes = len(params['R0'])

        df_wide = load_trajectory(str(traj_file))
        sample_data = (
            df_wide[df_wide['Sample'] == 0]
            .sort_values('t')
            .reset_index(drop=True)
        )

        xml_file = traj_file.with_name(
            traj_file.name.replace('_beast2.traj', '_beast2.xml')
        )
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file}")

        reaction_lookup = load_reactions(str(xml_file))
        species_cols = [c for c in df_wide.columns if c not in ('Sample', 't')]
        event_counts = classify_events(sample_data, reaction_lookup, species_cols)
        node_metrics = calculate_node_metrics(event_counts, num_nodes)
        peak_metrics = calculate_epidemic_peaks(sample_data, num_nodes)

        # --- Write merged output ---
        rows = [
            {
                'Location': n,
                'Initial_Population': params['Initial_Population'].get(n, 0),
                'Epidemic_Peak': peak_metrics[n]['peak'],
                'Peak_Timing': peak_metrics[n]['peak_time'],
                'Accumulated_Infections': node_metrics[n]['accumulated_infections'],
                'R0': params['R0'].get(n, 0.0),
                'Recovery_Rate': params['Recovery_Rate'].get(n, 0.0),
                'Source_Sink_Score': node_metrics[n]['source_sink_score'],
                'Ancestral_State': ancestral_labels.get(n, 0),
                'Spillover_Loc': params['Spillover_Loc'].get(n, 0),
            }
            for n in range(num_nodes)
        ]
        pd.DataFrame(rows).to_csv(output_file, index=False)
        return 'processed'

    except Exception as e:
        print(f"  Error processing {tree_file.name}: {e}")
        return 'error'


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: discover tree files and run process_simulation on each."""
    parser = argparse.ArgumentParser(
        description='Characterize outbreaks: extract node features and ancestral labels',
    )
    parser.add_argument('input_folder', type=Path, help='Directory containing BEAST2 outputs')
    parser.add_argument('--summary-file', type=Path, help='Write counts to file for bash aggregation')
    args = parser.parse_args()

    if not args.input_folder.exists():
        print(f"Error: Input folder not found: {args.input_folder}")
        sys.exit(1)

    tree_files = sorted(args.input_folder.glob('*_beast2.trees'))
    if not tree_files:
        print(f"No tree files found in {args.input_folder}")
        sys.exit(0)

    print(f"Characterizing outbreaks: {args.input_folder}")
    print(f"Found {len(tree_files)} simulations\n")

    processed = skipped = errors = 0
    for tree_file in tqdm(tree_files, desc="Processing"):
        prefix = tree_file.stem.replace('_beast2', '')
        traj_file = args.input_folder / f"{prefix}_beast2.traj"
        param_file = args.input_folder / f"{prefix}_parameter.csv"
        output_file = args.input_folder / f"{prefix}_nf.csv"

        if not traj_file.exists() or not param_file.exists():
            errors += 1
            continue

        result = process_simulation(tree_file, traj_file, param_file, output_file)
        if result == 'processed':
            processed += 1
        elif result == 'skipped':
            skipped += 1
        else:
            errors += 1

    print(f"\nProcessed: {processed} | Skipped: {skipped} | Errors: {errors}")

    if args.summary_file:
        with open(args.summary_file, 'w') as f:
            f.write(f"{processed} {skipped} {errors}\n")


if __name__ == '__main__':
    main()
