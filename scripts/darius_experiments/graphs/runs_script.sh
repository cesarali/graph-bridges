#!/bin/bash

python experiment_ctdd_graphs.py --lr 0.01
python experiment_ctdd_graphs.py --lr 0.0001 
python experiment_ctdd_graphs.py --lr 0.00001 

python experiment_ctdd_graphs.py --hidden-dim 64
python experiment_ctdd_graphs.py --hidden-dim 128
python experiment_ctdd_graphs.py --hidden-dim 512

python experiment_ctdd_graphs.py --ff-hidden-dim 64
python experiment_ctdd_graphs.py --ff-hidden-dim 128
python experiment_ctdd_graphs.py --ff-hidden-dim 256