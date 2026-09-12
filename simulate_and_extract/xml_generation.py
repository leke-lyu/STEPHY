#!/usr/bin/env python3
"""
BEAST2 XML Configuration File Generator

Generates random epidemic parameters (R0, recovery rate, migration, etc.)
and writes a BEAST2/ReMaster simulation XML configuration file.  Optionally
saves the drawn parameters to a companion CSV for downstream analysis.

All tuneable ranges are supplied via environment variables that
simulate_and_extract.sh exports before calling this script.

USAGE:
    python3 xml_generation.py <output.xml> [parameters.csv]

NOTE:
    This script requires all environment variables to be set by
    simulate_and_extract.sh.  It cannot be run standalone without them.
"""

import numpy as np
import pandas as pd
import sys
import os

# ============================================================
# CONFIGURATION
# ============================================================

def str_to_bool(s):
    """Convert string to boolean."""
    if isinstance(s, bool):
        return s
    return s.lower() in ('true', '1', 'yes')

def str_to_int_or_none(s):
    """Convert string to int, or None if the value is 'None'."""
    if s == "None" or s is None:
        return None
    return int(s)

def get_required_env(name):
    """Get required environment variable or raise ValueError."""
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Required environment variable '{name}' is not set. "
                        "This script must be called from 0_simulations.sh")
    return value

CONFIG = {
    'num_locs': int(get_required_env('NUM_LOCS')),
    'pop_range': (int(get_required_env('POP_MIN')), int(get_required_env('POP_MAX'))),
    'shared_pop_size': str_to_bool(get_required_env('SHARED_POP_SIZE')),
    'seed_location': str_to_int_or_none(get_required_env('SEED_LOCATION')),
    'R0_range': (float(get_required_env('R0_MIN')), float(get_required_env('R0_MAX'))),
    'shared_R0': str_to_bool(get_required_env('SHARED_R0')),
    'max_R0_diff': float(get_required_env('MAX_R0_DIFF')),
    'gamma_range': (float(get_required_env('GAMMA_MIN')), float(get_required_env('GAMMA_MAX'))),
    'shared_gamma': str_to_bool(get_required_env('SHARED_GAMMA')),
    'max_gamma_diff': float(get_required_env('MAX_GAMMA_DIFF')),
    'delta_range': (float(get_required_env('DELTA_MIN')), float(get_required_env('DELTA_MAX'))),
    'delta_hetero': float(os.getenv('DELTA_HETERO', '0')),
    'delta_hetero_time': float(os.getenv('DELTA_HETERO_TIME', '0')),
    'migration_range': (float(get_required_env('MIGRATION_MIN')), float(get_required_env('MIGRATION_MAX'))),
    'shared_migration_rate': str_to_bool(get_required_env('SHARED_MIGRATION_RATE')),
    'sim_time_range': (float(get_required_env('SIM_TIME_MIN')), float(get_required_env('SIM_TIME_MAX'))),
    'time_units': get_required_env('TIME_UNITS'),
    'ends_when': get_required_env('ENDS_WHEN'),
    'seed': str_to_int_or_none(get_required_env('RANDOM_SEED')),
    'num_sims': int(get_required_env('NUM_SIMS')),
}

# ============================================================
# PARAMETER GENERATION FUNCTIONS
# ============================================================

def generate_constrained_values(num_locs, value_range, max_diff):
    """Generate random values with constraint on maximum pairwise difference.

    Strategy: pick a random anchor within the feasible sub-range, then draw
    values within max_diff of that anchor.
    """
    # Calculate feasible range for the anchor point
    # The anchor must allow max_diff spread in both directions
    feasible_min = max(value_range[0], value_range[0] + max_diff / 2)
    feasible_max = min(value_range[1], value_range[1] - max_diff / 2)

    # If the constraint is impossible, use the full range
    if feasible_max < feasible_min:
        feasible_min = value_range[0]
        feasible_max = value_range[1]

    # Pick a random anchor point
    anchor = np.random.uniform(feasible_min, feasible_max)

    # Generate values within max_diff of the anchor
    constrained_min = max(value_range[0], anchor - max_diff / 2)
    constrained_max = min(value_range[1], anchor + max_diff / 2)

    values = np.random.uniform(constrained_min, constrained_max, size=num_locs)

    return values

def population_sizes(num_locs, pop_range, shared=False):
    """Generate population sizes for each location."""
    if shared:
        single_pop = np.random.uniform(pop_range[0], pop_range[1])
        pop_sizes = np.full(num_locs, single_pop)
    else:
        pop_sizes = np.random.uniform(pop_range[0], pop_range[1], size=num_locs)

    return pop_sizes.astype(int)

def seed_location(num_locs, seed_loc=None):
    """Generate one-hot encoded seed location (where epidemic starts)."""
    seed_array = np.zeros(num_locs, dtype=int)

    if seed_loc is None:
        selected_location = np.random.randint(0, num_locs)
    else:
        if not (0 <= seed_loc < num_locs):
            raise ValueError(f"seed_location must be between 0 and {num_locs-1}")
        selected_location = seed_loc

    seed_array[selected_location] = 1
    return seed_array

def R0_values(num_locs, R0_range, shared=False, max_diff=None):
    """Generate R0 values (basic reproduction number) for each location."""
    if shared:
        single_R0 = np.random.uniform(R0_range[0], R0_range[1])
        R0_array = np.full(num_locs, single_R0)
    else:
        R0_array = generate_constrained_values(num_locs, R0_range, max_diff)

    return R0_array

def recovery_rate(num_locs, gamma_range, shared=False, max_diff=None):
    """Generate recovery rates (γ) for each location."""
    if shared:
        single_gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        gamma_values = np.full(num_locs, single_gamma)
    else:
        gamma_values = generate_constrained_values(num_locs, gamma_range, max_diff)

    return gamma_values

def sample_rate(num_locs, delta_range, hetero=0.0, hetero_time=0.0):
    """Generate sampling rates. Returns (baseline x, matrix of shape
    (num_locs, num_phases)) — row i is location i, column k is time phase k.

    Default (both 0): one rate x ~ U(delta_range) shared by every location,
    constant in time -> shape (num_locs, 1).
    hetero = h > 0 (spatial): location i gets x * u_i, u_i ~ U(1-h, 1+h).
    hetero_time = h > 0 (temporal): every location shares three successive
    phases x*(1-h), x, x*(1+h) over equal thirds of the simulation time.
    In either case the baseline range is shrunk to [min/(1-h), max/(1+h)] so
    every rate still falls inside delta_range. The two modes are exclusive.
    """
    if hetero > 0 and hetero_time > 0:
        raise ValueError('DELTA_HETERO and DELTA_HETERO_TIME cannot both be set')
    lo, hi = delta_range
    h = max(hetero, hetero_time)
    x = np.random.uniform(lo / (1 - h), hi / (1 + h))
    if hetero > 0:
        return x, (x * np.random.uniform(1 - h, 1 + h, size=num_locs))[:, None]
    if hetero_time > 0:
        return x, np.tile(x * np.array([1 - h, 1.0, 1 + h]), (num_locs, 1))
    return x, np.full((num_locs, 1), x)

def phase_change_times(sim_time, num_phases):
    """Forward-time boundaries splitting sim_time into num_phases equal parts."""
    return [sim_time * k / num_phases for k in range(1, num_phases)]

def migration_rates(num_locs, migration_range, shared=False):
    """Generate migration rate matrix. Diagonal is 0 (no self-migration)."""
    migration_matrix = np.zeros((num_locs, num_locs))

    if shared:
        rate = np.random.uniform(migration_range[0], migration_range[1])
        for i in range(num_locs):
            for j in range(num_locs):
                if i != j:
                    migration_matrix[i, j] = rate
    else:
        for i in range(num_locs):
            for j in range(num_locs):
                if i != j:
                    migration_matrix[i, j] = np.random.uniform(migration_range[0], migration_range[1])

    return migration_matrix

def simulation_time(gamma_values, sim_time_range, time_units='recovery_period'):
    """Generate simulation time. Scales by 1/mean(γ) if time_units='recovery_period'."""
    raw_sim_time = np.random.uniform(sim_time_range[0], sim_time_range[1])

    if time_units == 'recovery_period':
        return raw_sim_time / np.mean(gamma_values)
    else:
        return raw_sim_time

# ============================================================
# DATA EXPORT
# ============================================================

def save_parameters_csv(pop_sizes, seed_number, R0_array, gamma_values, delta_value,
                        delta_values, beta_value, migration_rates_data, sim_time, output_file):
    """Save all generated epidemic parameters to a CSV file.

    delta_values / beta_value are (num_locs, num_phases) matrices. `sample_rate`
    always holds the baseline x. Spatial heterogeneity adds sample_rate_loc_{i};
    temporal heterogeneity adds sample_rate_phase_{k} (phase boundaries are
    simulation_time * k/num_phases). beta_loc_{i} is the phase where δ == x.
    """
    num_locs = len(pop_sizes)
    num_phases = delta_values.shape[1]
    ref_phase = int(np.argmin(np.abs(delta_values[0] - delta_value)))
    seed_loc_idx = np.argmax(seed_number)

    data = {}

    # Population sizes
    for i, pop in enumerate(pop_sizes):
        data[f'population_loc_{i}'] = [pop]

    # Seed location
    data['seed_location_index'] = [seed_loc_idx]
    for i, val in enumerate(seed_number):
        data[f'initial_infected_individuals_loc_{i}'] = [val]

    # R0 values
    for i, r0 in enumerate(R0_array):
        data[f'R0_loc_{i}'] = [r0]

    # Recovery rates
    for i, gamma in enumerate(gamma_values):
        data[f'recovery_rate_loc_{i}'] = [gamma]

    # Sample rate: baseline x, plus per-location / per-phase rates when heterogeneous
    data['sample_rate'] = [delta_value]
    if num_phases == 1 and not np.all(delta_values == delta_value):
        for i, delta in enumerate(delta_values[:, 0]):
            data[f'sample_rate_loc_{i}'] = [delta]
    if num_phases > 1:
        for k, delta in enumerate(delta_values[0]):
            data[f'sample_rate_phase_{k}'] = [delta]

    # Transmission rates (beta) at the baseline sampling rate
    for i, beta in enumerate(beta_value[:, ref_phase]):
        data[f'beta_loc_{i}'] = [beta]

    # Migration rates
    for i in range(num_locs):
        for j in range(num_locs):
            if i != j:
                data[f'migration_loc_{i}_to_loc_{j}'] = [migration_rates_data[i, j]]

    # Simulation time
    data['simulation_time'] = [sim_time]

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# ============================================================
# XML GENERATION
# ============================================================

def _rate_attr(rates, change_times):
    """Reaction rate attribute(s): a single rate, or a piecewise-constant list
    with the ReMaster changeTimes at which it switches."""
    rates = np.atleast_1d(rates)
    if len(rates) == 1:
        return f'rate="{rates[0]}"'
    return (f'rate="{" ".join(map(str, rates))}" '
            f'changeTimes="{" ".join(map(str, change_times))}"')

def generate_xml(pop_sizes, seed_number, beta_value, gamma_values, delta_values,
                 migration_rates_data, sim_time, num_sims, ends_when, output_file):
    """Generate BEAST2/ReMaster XML configuration file with SIR reactions.

    beta_value / delta_values are (num_locs, num_phases); with more than one
    phase the infection and sampling reactions switch rate at equal thirds
    of sim_time.
    """
    num_pops = len(pop_sizes)
    change_times = phase_change_times(sim_time, delta_values.shape[1])

    xml_lines = [
        '<beast version="2.0" namespace="beast.base.inference:beast.base.inference.parameter:remaster">',
        f'  <run spec="Simulator" nSims="{num_sims}">',
        '    <simulate id="tree" spec="SimulatedTree">',
        f'      <trajectory id="trajectory" spec="StochasticTrajectory" maxTime="{sim_time}" endsWhen="{ends_when}">',
        '      	',
        f'        <population spec="RealParameter" id="S" value="{" ".join(map(str, pop_sizes - seed_number))}"/>',
        f'        <population spec="RealParameter" id="I" value="{" ".join(map(str, seed_number))}"/>',
        f'        <population spec="RealParameter" id="R" value="0"/>',
        f'        <samplePopulation spec="RealParameter" id="sample" value="0"/>',
    ]

    # Infection reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" {_rate_attr(beta_value[i], change_times)}> S[{i}] + I[{i}] -> 2I[{i}] </reaction>')

    # Recovery reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" rate="{gamma_values[i]}"> I[{i}] -> R </reaction>')

    # Sampling reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" {_rate_attr(delta_values[i], change_times)}> I[{i}] -> R + sample </reaction>')

    # Migration reactions
    for i in range(num_pops):
        for j in range(num_pops):
            if i != j:
                xml_lines.append(f'        <reaction spec="Reaction" rate="{migration_rates_data[i, j]}"> I[{i}] -> I[{j}] </reaction>')

    xml_lines.extend([
        '',
        '      </trajectory>',
        '    </simulate>',
        '    <logger spec="Logger" fileName="$(filebase).traj">',
        '      <log idref="trajectory"/>',
        '    </logger>',
        '    <logger spec="Logger" mode="tree" fileName="$(filebase).trees">',
        '      <log spec="TypedTreeLogger" tree="@tree"/>',
        '    </logger>',
        '  </run>',
        '</beast>',
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(xml_lines))

# ============================================================
# MAIN EXECUTION
# ============================================================

def generate_parameters(config):
    """Generate all epidemic parameters based on configuration.

    Sets the random seed once; all downstream functions rely on this single
    seeding rather than managing their own RNG state.
    """
    if config['seed'] is not None:
        np.random.seed(config['seed'])

    # Generate parameters
    pop_sizes = population_sizes(
        config['num_locs'], config['pop_range'],
        config['shared_pop_size'],
    )

    seed_number = seed_location(
        config['num_locs'], config['seed_location'],
    )

    R0_array = R0_values(
        config['num_locs'], config['R0_range'],
        config['shared_R0'], config['max_R0_diff'],
    )

    gamma_values = recovery_rate(
        config['num_locs'], config['gamma_range'],
        config['shared_gamma'], config['max_gamma_diff'],
    )

    delta_value, delta_values = sample_rate(
        config['num_locs'], config['delta_range'],
        config['delta_hetero'], config['delta_hetero_time'],
    )

    # (num_locs, num_phases): beta follows delta phase by phase so R0 stays exact
    beta_value = (R0_array * (gamma_values[:, None] + delta_values).T / pop_sizes).T

    migration_rates_data = migration_rates(
        config['num_locs'], config['migration_range'],
        config['shared_migration_rate'],
    )

    sim_time = simulation_time(
        gamma_values, config['sim_time_range'],
        config['time_units'],
    )

    return pop_sizes, seed_number, R0_array, gamma_values, delta_value, delta_values, beta_value, migration_rates_data, sim_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 xml_Generation.py <output.xml> [parameters.csv]")
        sys.exit(1)

    output_xml = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) >= 3 else None

    # Generate parameters
    (pop_sizes, seed_number, R0_array, gamma_values, delta_value, delta_values,
     beta_value, migration_rates_data, sim_time) = generate_parameters(CONFIG)

    # Save CSV if requested
    if output_csv:
        save_parameters_csv(pop_sizes, seed_number, R0_array, gamma_values,
                           delta_value, delta_values, beta_value, migration_rates_data,
                           sim_time, output_csv)

    # Generate XML file
    generate_xml(pop_sizes, seed_number, beta_value, gamma_values, delta_values,
                 migration_rates_data, sim_time, CONFIG['num_sims'], CONFIG['ends_when'], output_xml)
