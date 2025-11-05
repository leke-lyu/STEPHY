#!/usr/bin/env python3
"""
Epidemic Parameter Generator and BEAST2 XML Configuration Tool

Generates random epidemic parameters and creates ReMaster simulation XML files.
All parameters MUST be provided via environment variables set by 0_simulations.sh.

USAGE:
    python3 xml_generation.py <output.xml> [parameters.csv]

NOTE:
    This script requires all environment variables to be set by 0_simulations.sh.
    It cannot be run standalone without these environment variables.
"""

import numpy as np
import pandas as pd
import sys
import os


# ============================================================
# CONFIGURATION
# ============================================================

def str_to_bool(s):
    """
    Convert string to boolean.

    Args:
        s: String or boolean value

    Returns:
        Boolean value
    """
    if isinstance(s, bool):
        return s
    return s.lower() in ('true', '1', 'yes')


def str_to_int_or_none(s):
    """
    Convert string to int or None.

    Args:
        s: String value or None

    Returns:
        Integer value or None if input is "None"
    """
    if s == "None" or s is None:
        return None
    return int(s)


def get_required_env(name):
    """
    Get required environment variable or raise error.

    Args:
        name: Name of the environment variable

    Returns:
        Value of the environment variable

    Raises:
        ValueError: If environment variable is not set
    """
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
    'mu_range': (float(get_required_env('MU_MIN')), float(get_required_env('MU_MAX'))),
    'shared_recovery_rate': str_to_bool(get_required_env('SHARED_RECOVERY_RATE')),
    'max_mu_diff': float(get_required_env('MAX_MU_DIFF')),
    'sample_range': (float(get_required_env('SAMPLE_MIN')), float(get_required_env('SAMPLE_MAX'))),
    'migration_range': (float(get_required_env('MIGRATION_MIN')), float(get_required_env('MIGRATION_MAX'))),
    'shared_migration_rate': str_to_bool(get_required_env('SHARED_MIGRATION_RATE')),
    'sim_time_range': (float(get_required_env('SIM_TIME_MIN')), float(get_required_env('SIM_TIME_MAX'))),
    'time_units': get_required_env('TIME_UNITS'),
    'seed': str_to_int_or_none(get_required_env('RANDOM_SEED')),
    'num_sims': int(get_required_env('NUM_SIMS')),
}


# ============================================================
# PARAMETER GENERATION FUNCTIONS
# ============================================================

def generate_constrained_values(num_locs, value_range, max_diff, seed=None):
    """
    Generate random values with constraint on maximum difference between values.

    Strategy:
        1. Calculate feasible range for an anchor point
        2. Pick random anchor within feasible range
        3. Generate values within max_diff of anchor
        This avoids bias toward the midpoint.

    Args:
        num_locs: Number of locations
        value_range: Tuple of (min, max) for allowed values
        max_diff: Maximum allowed difference between any two values
        seed: Random seed for reproducibility (optional)

    Returns:
        Array of constrained random values
    """
    if seed is not None:
        np.random.seed(seed)

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

def population_sizes(num_locs, pop_range, shared=False, seed=None):
    """
    Generate population sizes for each location.

    Args:
        num_locs: Number of locations
        pop_range: Tuple of (min, max) population size
        shared: If True, all locations have same population
        seed: Random seed for reproducibility (optional)

    Returns:
        Array of integer population sizes
    """
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_pop = np.random.uniform(pop_range[0], pop_range[1])
        pop_sizes = np.full(num_locs, single_pop)
    else:
        pop_sizes = np.random.uniform(pop_range[0], pop_range[1], size=num_locs)

    return pop_sizes.astype(int)


def seed_location(num_locs, seed_loc=None, seed=None):
    """
    Generate one-hot encoded seed location (where epidemic starts).

    Args:
        num_locs: Number of locations
        seed_loc: Specific location to seed (optional, if None chooses random)
        seed: Random seed for reproducibility (optional)

    Returns:
        One-hot encoded array indicating seed location

    Raises:
        ValueError: If seed_loc is out of bounds
    """
    if seed is not None:
        np.random.seed(seed)

    seed_array = np.zeros(num_locs, dtype=int)

    if seed_loc is None:
        selected_location = np.random.randint(0, num_locs)
    else:
        if not (0 <= seed_loc < num_locs):
            raise ValueError(f"seed_location must be between 0 and {num_locs-1}")
        selected_location = seed_loc

    seed_array[selected_location] = 1
    return seed_array


def R0_values(num_locs, R0_range, shared=False, max_diff=None, seed=None):
    """
    Generate R0 values (basic reproduction number) for locations.

    Args:
        num_locs: Number of locations
        R0_range: Tuple of (min, max) R0 values
        shared: If True, all locations have same R0
        max_diff: Maximum difference in R0 across locations (optional)
        seed: Random seed for reproducibility (optional)

    Returns:
        Array of R0 values
    """
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_R0 = np.random.uniform(R0_range[0], R0_range[1])
        R0_array = np.full(num_locs, single_R0)
    else:
        R0_array = generate_constrained_values(num_locs, R0_range, max_diff, seed)

    return R0_array


def recovery_rate(num_locs, mu_range, shared=False, max_diff=None, seed=None):
    """
    Generate recovery rates (mu) for locations.

    Args:
        num_locs: Number of locations
        mu_range: Tuple of (min, max) recovery rates
        shared: If True, all locations have same recovery rate
        max_diff: Maximum difference in recovery rates across locations (optional)
        seed: Random seed for reproducibility (optional)

    Returns:
        Array of recovery rates
    """
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_mu = np.random.uniform(mu_range[0], mu_range[1])
        mu_values = np.full(num_locs, single_mu)
    else:
        mu_values = generate_constrained_values(num_locs, mu_range, max_diff, seed)

    return mu_values


def sample_rate(sample_range, seed=None):
    """
    Generate sampling rate (single shared value for all locations).

    Args:
        sample_range: Tuple of (min, max) sampling rate
        seed: Random seed for reproducibility (optional)

    Returns:
        Single sampling rate value
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(sample_range[0], sample_range[1])


def migration_rates(num_locs, migration_range, shared=False, seed=None):
    """
    Generate migration rates between locations as a matrix.

    Args:
        num_locs: Number of locations
        migration_range: Tuple of (min, max) migration rates
        shared: If True, all location pairs have same migration rate
        seed: Random seed for reproducibility (optional)

    Returns:
        Matrix of migration rates (i,j) = rate from location i to j
        Diagonal elements are 0 (no self-migration)
    """
    if seed is not None:
        np.random.seed(seed)

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


def simulation_time(mu_values, sim_time_range, time_units='recovery_period', seed=None):
    """
    Generate simulation time.

    Args:
        mu_values: Array of recovery rates
        sim_time_range: Tuple of (min, max) simulation time
        time_units: 'recovery_period' (scaled by 1/mean(mu)) or 'arbitrary' (absolute)
        seed: Random seed for reproducibility (optional)

    Returns:
        Simulation time value
    """
    if seed is not None:
        np.random.seed(seed)

    raw_sim_time = np.random.uniform(sim_time_range[0], sim_time_range[1])

    if time_units == 'recovery_period':
        return raw_sim_time / np.mean(mu_values)
    else:
        return raw_sim_time


# ============================================================
# DATA EXPORT
# ============================================================

def save_parameters_csv(pop_sizes, seed_number, R0_array, mu_values, sample_rate_value,
                        beta_value, migration_rates_data, sim_time, output_file):
    """
    Save generated epidemic parameters to CSV file.

    Args:
        pop_sizes: Array of population sizes per location
        seed_number: One-hot encoded seed location
        R0_array: Array of R0 values per location
        mu_values: Array of recovery rates per location
        sample_rate_value: Sampling rate (shared across locations)
        beta_value: Array of transmission rates per location
        migration_rates_data: Matrix of migration rates between locations
        sim_time: Simulation time
        output_file: Path to output CSV file
    """
    num_locs = len(pop_sizes)
    seed_loc_idx = np.argmax(seed_number)

    data = {}

    # Population sizes
    for i, pop in enumerate(pop_sizes):
        data[f'population_loc_{i}'] = [pop]

    # Seed location
    data['seed_location_index'] = [seed_loc_idx]
    for i, val in enumerate(seed_number):
        data[f'seed_onehot_loc_{i}'] = [val]

    # R0 values
    for i, r0 in enumerate(R0_array):
        data[f'R0_loc_{i}'] = [r0]

    # Recovery rates
    for i, mu in enumerate(mu_values):
        data[f'recovery_rate_loc_{i}'] = [mu]

    # Sample rate
    data['sample_rate'] = [sample_rate_value]

    # Transmission rates (beta)
    for i, beta in enumerate(beta_value):
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

def generate_xml(pop_sizes, seed_number, beta_value, mu_values, sample_rate_value,
                 migration_rates_data, sim_time, num_sims, output_file):
    """
    Generate BEAST2/ReMaster XML configuration file.

    Args:
        pop_sizes: Array of population sizes per location
        seed_number: One-hot encoded seed location (initial infected per location)
        beta_value: Array of transmission rates per location
        mu_values: Array of recovery rates per location
        sample_rate_value: Sampling rate (shared across locations)
        migration_rates_data: Matrix of migration rates between locations
        sim_time: Maximum simulation time
        num_sims: Number of simulations to run
        output_file: Path to output XML file
    """
    num_pops = len(pop_sizes)

    xml_lines = [
        '<beast version="2.0" namespace="beast.base.inference:beast.base.inference.parameter:remaster">',
        f'  <run spec="Simulator" nSims="{num_sims}">',
        '    <simulate id="tree" spec="SimulatedTree">',
        f'      <trajectory id="trajectory" spec="StochasticTrajectory" maxTime="{sim_time}" mustHave="sample>=10">',
        '      	',
        f'        <population spec="RealParameter" id="S" value="{" ".join(map(str, pop_sizes))}"/>',
        f'        <population spec="RealParameter" id="I" value="{" ".join(map(str, seed_number))}"/>',
        f'        <population spec="RealParameter" id="R" value="0"/>',
        f'        <samplePopulation spec="RealParameter" id="sample" value="0"/>',
    ]

    # Infection reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" rate="{beta_value[i]}"> S[{i}] + I[{i}] -> 2I[{i}] </reaction>')

    # Recovery reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" rate="{mu_values[i]}"> I[{i}] -> R </reaction>')

    # Sampling reactions
    for i in range(num_pops):
        xml_lines.append(f'        <reaction spec="Reaction" rate="{sample_rate_value}"> I[{i}] -> R + sample </reaction>')

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
    """
    Generate all epidemic parameters based on configuration.

    Args:
        config: Configuration dictionary with all parameter ranges and settings

    Returns:
        Tuple of (pop_sizes, seed_number, R0_array, mu_values, sample_rate_value,
                  beta_value, migration_rates_data, sim_time)
    """
    # Generate parameters
    pop_sizes = population_sizes(
        config['num_locs'], config['pop_range'],
        config['shared_pop_size'], config['seed']
    )

    seed_number = seed_location(
        config['num_locs'], config['seed_location'], config['seed']
    )

    R0_array = R0_values(
        config['num_locs'], config['R0_range'],
        config['shared_R0'], config['max_R0_diff'], config['seed']
    )

    mu_values = recovery_rate(
        config['num_locs'], config['mu_range'],
        config['shared_recovery_rate'], config['max_mu_diff'], config['seed']
    )

    sample_rate_value = sample_rate(config['sample_range'], config['seed'])

    beta_value = R0_array * (mu_values + sample_rate_value) / pop_sizes

    migration_rates_data = migration_rates(
        config['num_locs'], config['migration_range'],
        config['shared_migration_rate'], config['seed']
    )

    sim_time = simulation_time(
        mu_values, config['sim_time_range'],
        config['time_units'], config['seed']
    )

    return pop_sizes, seed_number, R0_array, mu_values, sample_rate_value, beta_value, migration_rates_data, sim_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 xml_Generation.py <output.xml> [parameters.csv]")
        sys.exit(1)

    output_xml = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) >= 3 else None

    # Generate parameters
    pop_sizes, seed_number, R0_array, mu_values, sample_rate_value, beta_value, migration_rates_data, sim_time = generate_parameters(CONFIG)

    # Save CSV if requested
    if output_csv:
        save_parameters_csv(pop_sizes, seed_number, R0_array, mu_values,
                           sample_rate_value, beta_value, migration_rates_data,
                           sim_time, output_csv)

    # Generate XML file
    generate_xml(pop_sizes, seed_number, beta_value, mu_values, sample_rate_value,
                 migration_rates_data, sim_time, CONFIG['num_sims'], output_xml)
