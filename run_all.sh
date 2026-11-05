#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p data runs out
bash download_data.sh
python3 prepare_data.py data
python3 prepare_extra.py data
python3 precompute_shift.py
for s in 0 1 2 3 4; do python3 train.py $s 14; python3 features.py $s; done
python3 energy.py
python3 experiments.py 0,1,2,3,4 results
python3 stats.py
python3 diagrams.py
python3 plots.py
