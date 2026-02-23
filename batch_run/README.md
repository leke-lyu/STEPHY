srun -p interactive-cpu --pty bash

source ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate stephy

python3 outbreak_check.py /scratch/llyu30/epidata/30k/

Locations: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

=== All trees (30000 trees) ===
  tree_width     — Range: [160, 11472],  Mean: 2406.4
    Min: batch_1/1106_0 (160 tips)
    Max: batch_10/1043_0 (11472 tips)
  subtree_width  — Range: [11, 1732],  Mean: 240.6
    Min: tree=batch_0/1027_0, loc=6 (11 tips)
    Max: tree=batch_2/1251_0, loc=4 (1732 tips)

=== Discarding trees with any subtree_width > 872 (top 1% subtree threshold) ===
    30000 -> 28599 trees remain

  tree_width     — Range: [160, 7156],  Mean: 2209.1
    Min: batch_1/1106_0 (160 tips)
    Max: batch_0/534_0 (7156 tips)
  subtree_width  — Range: [11, 872],  Mean: 220.9
    Min: tree=batch_0/1027_0, loc=6 (11 tips)
    Max: tree=batch_1/1339_0, loc=4 (872 tips)

--num_locations 10 --subtree_width 872

bash submit_build_graphs.sh /scratch/llyu30/epidata/30k/ 872

python3 merge_graphs.py --data_dir /scratch/llyu30/epidata/28k/

bash submit_train.sh /scratch/llyu30/epidata/28k/graphs.pt 10

