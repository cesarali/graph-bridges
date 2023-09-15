#!/bin/bash

python experiment_ctdd_graphs.py --time-embed-dim 64 --cuda 3
python experiment_ctdd_graphs.py --time-embed-dim 256 --cuda 3
python experiment_ctdd_graphs.py --time-embed-dim 512 --cuda 3

python experiment_ctdd_graphs.py --num-layers 3 --cuda 3
python experiment_ctdd_graphs.py --num-layers 4 --cuda 3
python experiment_ctdd_graphs.py --num-layers 5 --cuda 3

python experiment_ctdd_graphs.py --num-heads 4 --cuda 3
python experiment_ctdd_graphs.py --num-heads 6 --cuda 3
python experiment_ctdd_graphs.py --num-heads 8 --cuda 3
