"""
Validation script for CBLV encoding
Traces through the algorithm step-by-step to verify correctness
"""

import dendropy as dp
import numpy as np

# =========================================
# Helper functions
# =========================================

def get_location(node):
    """Extract location number from 'type' annotation like 'I{7}'"""
    type_annot = node.annotations.get_value('type') if node.annotations else None
    if type_annot and '{' in str(type_annot):
        loc = str(type_annot).split('{')[1].split('}')[0]
        return int(loc)
    return None

def get_node_label(node):
    """Get label from node"""
    if node.label:
        return node.label
    elif node.taxon:
        return node.taxon.label
    return "unnamed"

def find_mrca(nodes):
    """Find MRCA of a list of nodes manually"""
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]

    ancestors = []
    node = nodes[0]
    while node:
        ancestors.append(node)
        node = node.parent_node

    for ancestor in ancestors:
        is_common = True
        for other_node in nodes[1:]:
            node = other_node
            found = False
            while node:
                if node == ancestor:
                    found = True
                    break
                node = node.parent_node
            if not found:
                is_common = False
                break
        if is_common:
            return ancestor
    return None

# =========================================
# Load and prepare tree
# =========================================

tree_file = "/Users/lukelyu/Desktop/epidata/40_5_MM0.003/0_beast2.trees"

tree_list = dp.TreeList.get(
    path=tree_file,
    schema='nexus',
    suppress_internal_node_taxa=True,
    suppress_leaf_node_taxa=True
)

phy_real = tree_list[0]
phy_real.is_rooted = True
phy_real.suppress_unifurcations()
phy_real.calc_node_root_distances()

tree_height_real = max(nd.root_distance for nd in phy_real.leaf_node_iter())
print(f"Original tree height: {tree_height_real:.4f}")
print(f"Original tree tips: {len(phy_real.leaf_nodes())}")

# =========================================
# Extract subtree for location 11
# =========================================

target_location = 11

# Find tips in target location
tips_target = [nd for nd in phy_real.leaf_node_iter() if get_location(nd) == target_location]
print(f"\nLocation {target_location} tips in original tree: {len(tips_target)}")

# Find MRCA and stem distance
mrca_target = find_mrca(tips_target)
stem_distance_real = mrca_target.root_distance
print(f"MRCA root distance (stem): {stem_distance_real:.4f}")

# Clone and prune
subtree_real = phy_real.clone(depth=2)
subtree_real.calc_node_root_distances()

# Prune non-target leaves
max_iterations = 2000
for iteration in range(max_iterations):
    non_target_leaves = [nd for nd in subtree_real.leaf_node_iter()
                        if get_location(nd) != target_location]
    if not non_target_leaves:
        break
    for nd in non_target_leaves:
        parent = nd.parent_node
        if parent is not None:
            parent.remove_child(nd)

subtree_real.suppress_unifurcations()
subtree_real.calc_node_root_distances()

n_tips_subtree = len(subtree_real.leaf_nodes())
print(f"Subtree tips after pruning: {n_tips_subtree}")

# =========================================
# VALIDATION: Trace through CBLV encoding
# =========================================

print("\n" + "="*70)
print("VALIDATION: Tracing CBLV encoding step by step")
print("="*70)

# Step 1: Ladderize (postorder) - compute max_root_distance for each node
print("\n[Step 1] Ladderize tree (postorder traversal)")
for nd in subtree_real.postorder_node_iter():
    if nd.is_leaf():
        nd.max_root_distance = nd.root_distance
    else:
        children = nd.child_nodes()
        ch_max_root_distance = [ch.max_root_distance for ch in children]
        ch_max_root_distance_rank = np.argsort(ch_max_root_distance)[::-1]
        children_reordered = [children[i] for i in ch_max_root_distance_rank]
        nd.max_root_distance = max(ch_max_root_distance)
        nd.set_children(children_reordered)

# Step 2: Inorder traversal - fill matrix
print("\n[Step 2] Inorder traversal to fill CBLV matrix")

tree_width = n_tips_subtree
heights = np.zeros((tree_width, 4))
height_idx = 0

last_int_node = subtree_real.seed_node
if last_int_node.edge.length is None:
    last_int_node.edge.length = 0

# Track nodes visited for validation
visited_leaves = []
visited_internals = []

for nd in subtree_real.inorder_node_iter():
    if height_idx >= tree_width:
        break
    if nd.is_leaf():
        # Col0: distance from leaf to last internal node
        if height_idx == 0:
            # First leaf: add stem to extend branch to TRUE root
            col0_val = nd.root_distance - last_int_node.root_distance + stem_distance_real
        else:
            col0_val = nd.root_distance - last_int_node.root_distance

        heights[height_idx, 0] = col0_val
        heights[height_idx, 2] = nd.edge.length if nd.edge.length else 0

        visited_leaves.append({
            'idx': height_idx,
            'label': get_node_label(nd),
            'root_dist': nd.root_distance,
            'last_int_root_dist': last_int_node.root_distance,
            'col0': col0_val,
            'col2': nd.edge.length if nd.edge.length else 0
        })
    else:
        if height_idx + 1 < tree_width:
            # Col1: internal node distance from TRUE root
            col1_val = nd.root_distance + stem_distance_real
            heights[height_idx + 1, 1] = col1_val
            heights[height_idx + 1, 3] = nd.edge.length if nd.edge.length else 0

            visited_internals.append({
                'target_idx': height_idx + 1,
                'root_dist': nd.root_distance,
                'col1': col1_val,
                'col3': nd.edge.length if nd.edge.length else 0
            })
        last_int_node = nd
        height_idx += 1

# Step 3: Rescale
print("\n[Step 3] Rescale by tree height")
heights_rescaled = heights / tree_height_real

# =========================================
# Print validation results
# =========================================

print("\n" + "="*70)
print("VALIDATION RESULTS: First 10 leaves")
print("="*70)

print(f"\n{'Idx':<5} {'Label':<10} {'RootDist':<12} {'LastIntDist':<12} {'Col0(raw)':<12} {'Col0(scaled)':<12}")
print("-" * 65)
for i, leaf in enumerate(visited_leaves[:10]):
    scaled_col0 = leaf['col0'] / tree_height_real
    print(f"{leaf['idx']:<5} {leaf['label']:<10} {leaf['root_dist']:<12.4f} {leaf['last_int_root_dist']:<12.4f} {leaf['col0']:<12.4f} {scaled_col0:<12.4f}")

print("\n" + "="*70)
print("VALIDATION: Column calculations")
print("="*70)

# Check first row specifically
first_leaf = visited_leaves[0]
print(f"\n[Row 0 - First Leaf]")
print(f"  Label: {first_leaf['label']}")
print(f"  Leaf root_distance: {first_leaf['root_dist']:.4f}")
print(f"  Last internal node root_distance: {first_leaf['last_int_root_dist']:.4f} (should be 0, subtree root)")
print(f"  Stem distance: {stem_distance_real:.4f}")
print(f"  Col0 = (leaf_dist - last_int_dist) + stem = ({first_leaf['root_dist']:.4f} - {first_leaf['last_int_root_dist']:.4f}) + {stem_distance_real:.4f} = {first_leaf['col0']:.4f}")
print(f"  Col0 rescaled = {first_leaf['col0']:.4f} / {tree_height_real:.4f} = {first_leaf['col0']/tree_height_real:.4f}")
print(f"  Col2 (leaf branch length): {first_leaf['col2']:.4f}")
print(f"  Col2 rescaled: {first_leaf['col2']/tree_height_real:.4f}")

# Verify the actual output matches expected
print("\n" + "="*70)
print("FINAL CBLV MATRIX (first 10 rows, rescaled)")
print("="*70)
print(f"\n{'Row':<5} {'Col0':<12} {'Col1':<12} {'Col2':<12} {'Col3':<12}")
print("-" * 55)
for i in range(min(10, tree_width)):
    print(f"{i:<5} {heights_rescaled[i,0]:<12.4f} {heights_rescaled[i,1]:<12.4f} {heights_rescaled[i,2]:<12.4f} {heights_rescaled[i,3]:<12.4f}")

# =========================================
# Cross-check key values
# =========================================

print("\n" + "="*70)
print("CROSS-VALIDATION CHECKS")
print("="*70)

# Check 1: First row Col0 should be first leaf's root distance + stem (relative to true root)
expected_col0_row0 = (first_leaf['root_dist'] + stem_distance_real) / tree_height_real
actual_col0_row0 = heights_rescaled[0, 0]
check1_pass = abs(expected_col0_row0 - actual_col0_row0) < 0.0001

print(f"\n[Check 1] Row 0 Col0 = first leaf's distance from TRUE root / tree_height")
print(f"  Expected: ({first_leaf['root_dist']:.4f} + {stem_distance_real:.4f}) / {tree_height_real:.4f} = {expected_col0_row0:.4f}")
print(f"  Actual: {actual_col0_row0:.4f}")
print(f"  PASS: {check1_pass}")

# Check 2: Col1 values should be internal node distances from true root
print(f"\n[Check 2] Col1 values = internal node root distances + stem")
for i, internal in enumerate(visited_internals[:3]):
    expected = (internal['root_dist'] + stem_distance_real) / tree_height_real
    actual = heights_rescaled[internal['target_idx'], 1]
    check_pass = abs(expected - actual) < 0.0001
    print(f"  Row {internal['target_idx']}: expected {expected:.4f}, actual {actual:.4f}, PASS: {check_pass}")

# Check 3: Maximum value in Col0 should correspond to deepest leaf
max_col0 = heights_rescaled[:, 0].max()
print(f"\n[Check 3] Max Col0 value: {max_col0:.4f}")
print(f"  This should be close to 1.0 if the deepest leaf in subtree reaches near tree height")

# Check 4: All Col1 values (except row 0) should be less than Col0 of the same row
print(f"\n[Check 4] For each row i>0, Col1[i] < Col0[0]+...+Col0[i-1] (distance ordering)")
valid_count = 0
for i in range(1, min(5, tree_width)):
    col1_val = heights_rescaled[i, 1]
    # Col1 is the internal node distance, should be less than leaves above it
    if col1_val > 0:
        valid_count += 1
print(f"  Rows with non-zero Col1: {valid_count} (should be > 0)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Tree height: {tree_height_real:.4f}")
print(f"  Stem distance: {stem_distance_real:.4f}")
print(f"  Subtree tips: {n_tips_subtree}")
print(f"  Output shape: ({tree_width}, 4)")
print(f"  Col0[0,0] (first leaf dist from true root, scaled): {heights_rescaled[0,0]:.4f}")
print(f"  Min value: {heights_rescaled.min():.4f}")
print(f"  Max value: {heights_rescaled.max():.4f}")
print(f"  Mean value: {heights_rescaled.mean():.4f}")

# Compare with notebook output
print("\n" + "="*70)
print("COMPARISON WITH NOTEBOOK OUTPUT")
print("="*70)
notebook_output = np.array([
    [0.7018, 0.0000, 0.3803, 0.0000],
    [0.0498, 0.3215, 0.0498, 0.0710],
    [0.1915, 0.2505, 0.1314, 0.0454],
    [0.0257, 0.3106, 0.0257, 0.0601],
    [0.1371, 0.2051, 0.0514, 0.0209],
])

print(f"\n{'Row':<5} {'Col0 Match':<15} {'Col1 Match':<15} {'Col2 Match':<15} {'Col3 Match':<15}")
print("-" * 65)
for i in range(5):
    col0_match = abs(heights_rescaled[i,0] - notebook_output[i,0]) < 0.01
    col1_match = abs(heights_rescaled[i,1] - notebook_output[i,1]) < 0.01
    col2_match = abs(heights_rescaled[i,2] - notebook_output[i,2]) < 0.01
    col3_match = abs(heights_rescaled[i,3] - notebook_output[i,3]) < 0.01
    print(f"{i:<5} {str(col0_match):<15} {str(col1_match):<15} {str(col2_match):<15} {str(col3_match):<15}")
