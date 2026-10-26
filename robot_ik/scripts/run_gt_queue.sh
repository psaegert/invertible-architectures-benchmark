#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Generate all FR3 ground-truth sets one after another on one GPU.
#
#   nohup robot_ik/scripts/run_gt_queue.sh GPU > gt_queue.log 2>&1 < /dev/null &
#
# Run from the repository root with the venv in .venv. Pick a GPU that no one
# else is using (nvidia-smi --query-compute-apps); thread counts are capped so
# the job does not crowd a shared machine.
set -u
GPU=${1:?usage: run_gt_queue.sh GPU}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for job in "fr3_6d --split val" "fr3_5d --split val" "fr3_6d --split test" "fr3_5d --split test"; do
  echo "$(date -Is) start $job"
  .venv/bin/python -m robot_ik.scripts.generate_gt $job --device cuda > "gt_${job// /_}.log" 2>&1
  echo "$(date -Is) exit $? $job"
done
