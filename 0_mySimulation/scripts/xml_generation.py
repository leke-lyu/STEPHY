#!/usr/bin/env python3
"""
Epidemic Parameter Generator and BEAST2 XML Configuration Tool

Generates random epidemic parameters and creates ReMaster simulation XML files.
Parameters are read from environment variables set by 00_simulations.sh.

USAGE:
    python3 xml_Generation.py <output.xml> [parameters.csv]
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
    """Convert string to int or None."""
    if s == "None" or s is None:
        return None
    return int(s)

CONFIG = {
    'num_locs': int(os.getenv('NUM_LOCS', 10)),
    'pop_range': (int(os.getenv('POP_MIN', 1000)), int(os.getenv('POP_MAX', 10000))),
    'shared_pop_size': str_to_bool(os.getenv('SHARED_POP_SIZE', 'False')),
    'seed_location': str_to_int_or_none(os.getenv('SEED_LOCATION', 'None')),
    'R0_range': (float(os.getenv('R0_MIN', 2)), float(os.getenv('R0_MAX', 5))),
    'shared_R0': str_to_bool(os.getenv('SHARED_R0', 'False')),
    'max_R0_diff': float(os.getenv('MAX_R0_DIFF', 1)),
    'mu_range': (float(os.getenv('MU_MIN', 0.02)), float(os.getenv('MU_MAX', 0.05))),
    'shared_recovery_rate': str_to_bool(os.getenv('SHARED_RECOVERY_RATE', 'False')),
    'max_mu_diff': float(os.getenv('MAX_MU_DIFF', 0.01)),
    'sample_range': (float(os.getenv('SAMPLE_MIN', 0.0001)), float(os.getenv('SAMPLE_MAX', 0.0002))),
    'migration_range': (float(os.getenv('MIGRATION_MIN', 0.0001)), float(os.getenv('MIGRATION_MAX', 0.002))),
    'shared_migration_rate': str_to_bool(os.getenv('SHARED_MIGRATION_RATE', 'False')),
    'sim_time_range': (float(os.getenv('SIM_TIME_MIN', 4)), float(os.getenv('SIM_TIME_MAX', 8))),
    'time_units': os.getenv('TIME_UNITS', 'recovery_period'),
    'seed': str_to_int_or_none(os.getenv('RANDOM_SEED', 'None')),
    'num_sims': int(os.getenv('NUM_SIMS', 1)),
}


# ============================================================
# PARAMETER GENERATION FUNCTIONS
# ============================================================

def generate_constrained_values(num_locs, value_range, max_diff, seed=None):
    """
    Generate values with constraint on maximum difference.

    Strategy: Pick a random anchor point within the feasible range,
    then generate other values within max_diff of that anchor.
    This avoids bias toward the midpoint.
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
    """Generate population sizes for each location."""
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_pop = np.random.uniform(pop_range[0], pop_range[1])
        pop_sizes = np.full(num_locs, single_pop)
    else:
        pop_sizes = np.random.uniform(pop_range[0], pop_range[1], size=num_locs)

    return pop_sizes.astype(int)

def seed_location(num_locs, seed_loc=None, seed=None):
    """Generate one-hot encoded seed location (where epidemic starts)."""
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
    """Generate R0 values (basic reproduction number) for locations."""
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_R0 = np.random.uniform(R0_range[0], R0_range[1])
        R0_array = np.full(num_locs, single_R0)
    else:
        R0_array = generate_constrained_values(num_locs, R0_range, max_diff, seed)

    return R0_array

def recovery_rate(num_locs, mu_range, shared=False, max_diff=None, seed=None):
    """Generate recovery rate(s) (mu) for locations."""
    if seed is not None:
        np.random.seed(seed)

    if shared:
        single_mu = np.random.uniform(mu_range[0], mu_range[1])
        mu_values = np.full(num_locs, single_mu)
    else:
        mu_values = generate_constrained_values(num_locs, mu_range, max_diff, seed)

    return mu_values

def sample_rate(sample_range, seed=None):
    """Generate sample rate (single shared value for all locations)."""
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(sample_range[0], sample_range[1])

def migration_rates(num_locs, migration_range, shared=False, seed=None):
    """Generate migration rates between locations as a matrix."""
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
    If time_units='recovery_period', scales by mean recovery period (1/mean(mu)).
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
    """Save generated parameters to CSV file."""
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
    """Generate BEAST2 XML configuration file."""
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
    """Generate all epidemic parameters based on configuration."""
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
