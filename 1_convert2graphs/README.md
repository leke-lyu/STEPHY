# Graph Feature Extraction and Visualization

This directory contains tools for converting BEAST2 simulation outputs into graph features and visualizing outbreak dynamics.

## Files

### Core Scripts
- **`node_feature.py`**: Extracts node-level features for GNN analysis
  - Input: XML + trajectory files
  - Output: CSV with node features (Initial_Population, Accumulated_Infections, Num_Samples, Source_Sink_Score)

- **`edge_feature.R`**: Extracts edge/migration features between locations

- **`trajectory_utils.py`**: Shared utility functions for trajectory processing
  - Used by both `node_feature.py` and visualization notebooks
  - Provides data loading, parsing, and organization functions

### Visualization
- **`visualize_outbreak.ipynb`**: Jupyter notebook for interactive outbreak visualization
  - Plots compartment dynamics (S, I, R) over time
  - Analyzes peak infections by location
  - Compares multiple simulation samples

## Usage

### Extract Node Features
```bash
python3 node_feature.py <xml_file> <traj_file> [output.csv]

# Example:
python3 node_feature.py 0_beast2.xml 0_beast2.traj 0_node.csv
```

### Visualize Outbreak Dynamics
```bash
jupyter notebook visualize_outbreak.ipynb
```

Then configure the file paths in the notebook:
```python
TRAJ_FILE = '0_beast2.traj'
XML_FILE = '0_beast2.xml'
SAMPLE_ID = 0
```

### Use Utility Functions Programmatically
```python
from trajectory_utils import load_trajectory_wide, get_compartment_groups

# Load trajectory data
df = load_trajectory_wide('0_beast2.traj')

# Organize compartments
groups = get_compartment_groups(df)
print(groups['I'])  # ['I[0]', 'I[1]', 'I[2]', ...]

# Plot infected compartments
import matplotlib.pyplot as plt
sample_0 = df[df['Sample'] == 0]
sample_0.plot(x='t', y=groups['I'], figsize=(12, 6))
plt.show()
```

## Shared Utility Functions

The `trajectory_utils.py` module provides reusable functions:

### Data Loading
- `load_trajectory_wide(traj_file)`: Load trajectory in wide format (for plotting)
- `load_trajectory_long(traj_file)`: Load trajectory in long format (original)
- `load_reactions_from_xml(xml_file)`: Extract reaction definitions
- `load_parameters_from_xml(xml_file)`: Extract simulation parameters

### Data Organization
- `create_species_name(pop, idx)`: Create standardized species names
- `get_compartment_groups(df_wide)`: Organize species into S, I, R groups
- `get_sample_data(df_wide, sample_id)`: Extract specific simulation sample
- `get_trajectory_summary(traj_file)`: Get quick summary statistics

### Reaction Parsing
- `parse_reaction(reaction_str)`: Parse reaction strings into components

## Output Formats

### Node Features CSV
```
graph_id,node,Initial_Population,Accumulated_Infections,Num_Samples,Source_Sink_Score
0_0,0,3329,3453,24,-0.137
0_0,1,3869,4284,23,0.095
...
```

### Trajectory Data (Wide Format)
```
Sample  t      S[0]    S[1]    I[0]    I[1]    R    sample
0       0.0    3329    3869    0       0       0    0
0       0.5    3328    3869    1       0       0    0
...
```

## Visualization Examples

The notebook provides 7 types of visualizations:

1. **Total Population Dynamics**: S, I, R totals over time
2. **Infected by Location**: Individual I[x] trajectories
3. **Susceptible by Location**: Individual S[x] trajectories
4. **Infection Heatmap**: Spatial-temporal infection patterns
5. **Stacked Area Plot**: Compartment distribution
6. **Peak Infection Analysis**: Peak timing and magnitude by location
7. **Multi-Sample Comparison**: Compare trajectories across simulations

## Dependencies

```bash
# Python packages
pip install pandas numpy matplotlib seaborn jupyter

# R packages (for edge_feature.R)
install.packages(c("data.table", "ape"))
```

## Tips

- For large trajectory files (>100MB), loading may take 1-2 minutes
- The `load_trajectory_wide()` function is optimized for plotting
- Use `get_sample_data()` to work with individual simulations
- The utility module can be imported in any Python script or notebook

## Troubleshooting

**Issue**: File not found errors
**Solution**: Use absolute paths or ensure working directory is correct

**Issue**: Memory errors with large files
**Solution**: Process one sample at a time using `get_sample_data()`

**Issue**: Import errors in notebook
**Solution**: Ensure `trajectory_utils.py` is in the same directory or in Python path
