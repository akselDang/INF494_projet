#!/usr/bin/env bash
# All MNIST experiments, 3 seeds each, then the summary table and the figures.
set -e
for seed in 0 1 2; do
  python -m src.train --dataset mnist --model gcn --features 3 --seed "$seed"
  python -m src.train --dataset mnist --model gcn  --seed "$seed"
  python -m src.train --dataset mnist --model sage --seed "$seed"
  python -m src.train --dataset mnist --model sage --raw-features --seed "$seed"
  python -m src.train --dataset mnist --model lenet --seed "$seed"
  python -m src.train --dataset mnist --model cnn  --seed "$seed"
done
python -m src.summarize
python report/make_numbers.py
python -m src.visualize --dataset mnist --models cnn lenet sage gcn --seed 0
python -m src.demo_pipeline --stats

# CIFAR-10 (optional, longer):
# for seed in 0 1 2; do
#   python -m src.train --dataset cifar10 --model sage --seed "$seed"
#   python -m src.train --dataset cifar10 --model cnn  --seed "$seed"
# done
# python -m src.summarize
