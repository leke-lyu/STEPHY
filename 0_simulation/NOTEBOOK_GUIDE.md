# Visualization Notebook Quick Guide

## ✅ Both Issues Fixed!

### Issue 1: Missing seaborn ✓
**Status:** INSTALLED
```bash
pip3 install seaborn  # Already done!
```

### Issue 2: Import trajectory_utils from new location ✓
**Status:** FIXED in the notebook

The first cell now automatically finds and imports utilities from `../utils/`:

```python
import sys
from pathlib import Path
import os

# Automatically find utils directory
notebook_dir = Path(os.getcwd())
utils_dir = notebook_dir.parent / 'utils'
sys.path.insert(0, str(utils_dir))

# Import all utility functions
from trajectory_utils import (
    load_trajectory_wide,
    load_parameters_from_xml,
    get_compartment_groups,
    get_sample_data,
    get_trajectory_summary
)
```

## How to Use

### 1. Start Jupyter Notebook
```bash
cd /Users/lukelyu/Library/CloudStorage/OneDrive-Emory/emory/phyloGNN/0_simulation
jupyter notebook visualize_outbreak.ipynb
```

### 2. Run the First Cell
The first cell will:
- Import all required packages (pandas, numpy, matplotlib, seaborn)
- Automatically locate and import utilities from `../utils/trajectory_utils.py`
- Set up plotting styles
- Print confirmation messages

You should see:
```
✓ Imports successful!
✓ Utils loaded from: /Users/.../phyloGNN/utils
```

### 3. Configure Your Data Files
In the second cell, set these paths:
```python
TRAJ_FILE = '0_beast2.traj'   # Your trajectory file
XML_FILE = '0_beast2.xml'     # Your XML file
SAMPLE_ID = 0                  # Which sample to visualize
```

### 4. Run All Cells
Click "Cell" → "Run All" or use Shift+Enter to run each cell.

## What Functions Are Available?

From `trajectory_utils`, you get:

### Data Loading
- `load_trajectory_wide(file)` - Load trajectory for plotting
- `load_parameters_from_xml(file)` - Get simulation parameters
- `get_trajectory_summary(file)` - Quick statistics

### Data Organization
- `get_compartment_groups(df)` - Organize columns into S, I, R, sample
- `get_sample_data(df, id)` - Extract specific simulation sample

### All imported automatically! ✓

## Troubleshooting

### If imports fail:
Check that utils directory exists:
```bash
ls /Users/lukelyu/Library/CloudStorage/OneDrive-Emory/emory/phyloGNN/utils/
# Should show: trajectory_utils.py
```

### If seaborn is missing:
```bash
pip3 install seaborn
```

### If files not found:
Make sure you're in the correct directory when running Jupyter:
```bash
pwd  # Should show: .../phyloGNN/0_simulation
```

## Folder Structure

```
phyloGNN/
├── 0_simulation/
│   ├── visualize_outbreak.ipynb  ← You are here
│   └── NOTEBOOK_GUIDE.md         ← This file
│
├── utils/
│   └── trajectory_utils.py       ← Imported automatically
│
└── 1_convert2graphs/
    └── node_feature.py
```

## Quick Test

After running the import cell, test it works:
```python
# This should work without errors
from trajectory_utils import load_trajectory_wide
print("✓ Imports working!")
```

---

**Everything is now configured and ready to use! 🎉**
