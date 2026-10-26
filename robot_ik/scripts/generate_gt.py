# SPDX-License-Identifier: MIT
"""Generate the exact ground truth for the committed FR3 conditions.

    python -m robot_ik.scripts.generate_gt fr3_6d --split test [--shard 0 --n-shards 4] [--device cuda]
    python -m robot_ik.scripts.generate_gt fr3_6d --split test --merge --n-shards 4

Without --n-shards it writes the cache file metrics.evaluate reads directly
(metrics/cache/gt_fr3_..., keyed by y*, N_GT and GT_SEED like the toy ground
truth). With shards, each shard covers a contiguous block of poses and writes
metrics/cache/shards/; --merge joins them into the cache file. Diagnostics
are printed and stored in the file (robot_ik/ground_truth.py).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from metrics.benchmarks import load_test_conditions
from metrics.ground_truth import GT_SEED, N_GT
from robot_ik.ground_truth import CACHE_DIR, N_SEEDS, cache_path, generate, load, save


def shard_path(task, split, k, n):
    return CACHE_DIR / 'shards' / f'{task}_{split}_shard{k}of{n}.npz'


def summary(diag):
    loops = diag['n_loops']
    lines = [f"poses {len(loops)}; loops per pose min {loops.min()} median {int(np.median(loops))} max {loops.max()}",
             f"all closed: {bool(diag['all_closed'].all())}; max closure error {np.nanmax(diag['max_closure_error']):.1e}",
             f"in-box share of the fiber mass: min {diag['in_box_share'].min():.3f} median {np.median(diag['in_box_share']):.3f}"]
    if 'x_true_on_fiber' in diag:
        miss = int((~diag['x_true_on_fiber'].astype(bool)).sum())
        lines.append(f"x_true not on a traced loop: {miss} of {len(loops)}")
    return '\n'.join('  ' + s for s in lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('benchmark', choices=['fr3_6d', 'fr3_5d'])
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--n-seeds', type=int, default=N_SEEDS)
    args = p.parse_args()
    torch.set_num_threads(4)
    task = args.benchmark[4:]
    y_star, x_true = load_test_conditions(args.benchmark, args.split)
    out = cache_path(task, y_star, N_GT, GT_SEED)

    if args.merge:
        parts = [load(shard_path(task, args.split, k, args.n_shards)) for k in range(args.n_shards)]
        draws = [sum((pt[0] for pt in parts), []), sum((pt[1] for pt in parts), [])]
        diag = {k: np.concatenate([np.atleast_1d(pt[2][k]) for pt in parts]) if np.ndim(parts[0][2][k])
                else parts[0][2][k] for k in parts[0][2]}
        save(out, draws, y_star, diag)
        print(f'merged {args.n_shards} shards -> {out}\n{summary(diag)}')
        return

    blocks = np.array_split(np.arange(len(y_star)), args.n_shards)
    sel = blocks[args.shard]
    t0 = time.time()
    draws, diag = generate(task, y_star[sel], n_gt=N_GT, seed=GT_SEED * 1000 + args.shard, n_seeds=args.n_seeds,
                           device=args.device, x_true=x_true[sel])
    target = out if args.n_shards == 1 else shard_path(task, args.split, args.shard, args.n_shards)
    save(target, draws, y_star[sel], diag)
    print(f'{args.benchmark} {args.split} shard {args.shard}/{args.n_shards}: {len(sel)} poses in '
          f'{time.time() - t0:.0f}s -> {target}\n{summary(diag)}', flush=True)


if __name__ == '__main__':
    main()
