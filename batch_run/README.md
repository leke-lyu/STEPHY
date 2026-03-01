# Batch Run Pipeline

End-to-end workflow: check data, build per-batch graphs, merge, train, and test.

## Prerequisites

```bash
srun -p interactive-cpu --pty bash
source ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate stephy
```

## 1. Check data and determine parameters

```bash
python3 outbreak_check.py <data_dir>
```

Note the recommended `--num_locations` and `--subtree_width` from the output.

## 2. Build per-batch graphs (SLURM array)

```bash
bash submit_build_graphs.sh <data_dir> <subtree_width>
```

Reads from `<data_dir>/batch_*/` and writes each batch's graphs to
`<output_dir>/batch_*_graphs.pt`. Logs go to `<output_dir>/logs/`.

## 3. Merge batch graphs

```bash
python3 merge_graphs.py --result_dir <output_dir>
```

Merges all `batch_*_graphs.pt` files into a single `graphs.pt`.

## 4. Train (SLURM array, 3 pipelines x 4 labels = 12 jobs)

```bash
bash submit_train.sh <output_dir>/graphs.pt <num_locations>
```

Trained models are saved to `<output_dir>/<pipeline>/<label>/`.

## 5. Test on a new dataset

```bash
python3 test.py --graphs <graphs.pt> --model_dir <model_dir> --num_locations <N>
```

Results are written to the same directory as `--graphs`.

## Output structure

```
<data_dir>/                     # input data (untouched)
  batch_0/*.trees, *.csv
  batch_1/...

<output_dir>/                   # all outputs
  batch_0_graphs.pt
  batch_1_graphs.pt
  graphs.pt                     # merged
  logs/                         # SLURM logs
  stephy2/r0/                   # training results
  stephy2/rr/
  CBLV-CNN2/...
  CBLV-GAT2/...
```
