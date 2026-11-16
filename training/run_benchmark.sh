#!/usr/bin/env bash
# Full benchmark on TN5000 (+ Kaggle external test): FibonacciNet vs a standard CNN, 3 seeds each, with ONE shared
# training recipe, then the evaluation report.
#
# Recipe: selected on the Kaggle validation split only (reports/sweep.json): cosine schedule,
# peak LR 2e-3, standard augmentation. 60 epochs there = ~3,700 optimizer steps; 40 epochs on the
# larger TN5000 train split = ~4,500 steps, so the optimization budget is matched.
# Override with TRAIN_ARGS. Finished runs are skipped, so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-mac_env/bin/python}
TRAIN_ARGS=${TRAIN_ARGS:-"--schedule cosine --lr 2e-3 --aug standard --epochs 40"}

$PY -m training.dataset --dataset kaggle
$PY -m training.dataset --dataset tn5000
mkdir -p logs/runs

run_phase() {
  for seed in 0 1 2; do
    for model in "$@"; do
      if [[ -f runs/tn5000/$model/seed$seed/preds.npz ]]; then
        echo "skip $model seed$seed"; continue
      fi
      # shellcheck disable=SC2086
      $PY -m training.train --dataset tn5000 --model "$model" --seed "$seed" $TRAIN_ARGS 2>&1 \
        | tee "logs/runs/${model}_seed${seed}.log"
    done
  done
}

run_phase fibonacci_net baseline_cnn
$PY -m training.evaluate
