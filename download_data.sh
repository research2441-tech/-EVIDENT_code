#!/bin/bash
set -e
mkdir -p data && cd data
B=https://huggingface.co/datasets/bazyl/GTSRB/resolve/main
curl -L -o gtsrb_Train.csv $B/Train.csv
curl -L -o gtsrb_Test.csv $B/Test.csv
curl -L -o gtsrb_train-00000-of-00001-dc762c064c221993.parquet $B/data/train-00000-of-00001-dc762c064c221993.parquet
curl -L -o gtsrb_test-00000-of-00001-747a54d4a6461a97.parquet $B/data/test-00000-of-00001-747a54d4a6461a97.parquet
curl -L -o indian.zip https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification/resolve/main/Indian-Traffic-Sign-Classification.zip
