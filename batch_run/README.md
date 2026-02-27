srun -p interactive-cpu --pty bash

source ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate stephy

python3 outbreak_check.py /projects/lau_projects/epidata/100k

(stephy) [llyu30@node23 batch_run]$  python3 outbreak_check.py /projects/lau_projects/epidata/100k
Dataset:   /projects/lau_projects/epidata/100k
Locations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

=== All trees (100000 trees) ===
  tree_width     — Range: [202, 11001],  Mean: 2341.3
    Min: batch_23/2418_0 (202 tips)
    Max: batch_3/2568_0 (11001 tips)
  subtree_width  — Range: [11, 1665],  Mean: 234.1
    Min: tree=batch_0/1211_0, loc=8 (11 tips)
    Max: tree=batch_22/2279_0, loc=4 (1665 tips)

=== Discarding trees with any subtree_width > 834 (top 1% subtree threshold) ===
    100000 -> 95157 trees remain

  tree_width     — Range: [202, 6845],  Mean: 2150.9
    Min: batch_23/2418_0 (202 tips)
    Max: batch_2/2504_0 (6845 tips)
  subtree_width  — Range: [11, 834],  Mean: 215.1
    Min: tree=batch_0/1211_0, loc=8 (11 tips)
    Max: tree=batch_1/3894_0, loc=5 (834 tips)

--num_locations 10 --subtree_width 834

bash submit_build_graphs.sh /projects/lau_projects/epidata/100k 834

python3 merge_graphs.py --data_dir /projects/lau_projects/epidata/100k

bash submit_train.sh /projects/lau_projects/epidata/100k/graphs.pt 10

