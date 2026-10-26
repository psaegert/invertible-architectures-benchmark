# SPDX-License-Identifier: MIT
"""Checks of the FR3 ground truth (results are recorded in docs/robot_ik.md).

    python -m robot_ik.scripts.validate_gt report fr3_6d [--split test]
        diagnostics of the cached set; simulation-based calibration of the
        ground truth itself against the configurations that generated y*;
        the oracle row (the second, independent draw scored like a model),
        including its coverage at several radii.
    python -m robot_ik.scripts.validate_gt refine fr3_6d [--n-poses 20]
        regenerate the first validation poses with twice the seeds and half the
        step; the per-pose loop count and mass must not change.
    python -m robot_ik.scripts.validate_gt direct [--n-poses 10]
        6-D task only: trace the fiber directly in all 7 joints (6 constraints,
        geometric Jacobian) instead of through the joint 1-6 curve, and compare
        the two posteriors per pose with an MMD permutation test.
"""

import argparse
import json

import numpy as np
import torch

from metrics.benchmarks import load_test_conditions
from metrics.calibration import calibration_report
from metrics.evaluate import evaluate
from metrics.ground_truth import GT_SEED, N_GT
from metrics.mmd import imq_kernel, sq_dists
from robot_ik import kinematics as K
from robot_ik.ground_truth import H_MAX, N_SEEDS, cache_path, generate, load, oracle_sampler
from robot_ik.manifold import FiberProblem, discover_and_trace, into_box, sample_loops
from robot_ik.metrics import coverage

DELTAS = (0.05, 0.1, 0.2, 0.5, 1.0)


def report(benchmark, split):
    task = benchmark[4:]
    y, x_true = load_test_conditions(benchmark, split)
    gt, oracle, diag = load(cache_path(task, y, N_GT, GT_SEED))
    out = {'benchmark': benchmark, 'split': split, 'n_poses': len(y),
           'loops_per_pose': {'min': int(diag['n_loops'].min()), 'median': float(np.median(diag['n_loops'])),
                              'max': int(diag['n_loops'].max())},
           'all_closed': bool(diag['all_closed'].all()),
           'max_closure_error': float(np.nanmax(diag['max_closure_error'])),
           'in_box_share_median': float(np.median(diag['in_box_share'])),
           'x_true_not_on_fiber': int((~diag['x_true_on_fiber'].astype(bool)).sum())}
    sbc = calibration_report(np.stack(gt), x_true)
    out['ground_truth_sbc'] = {'marginal': sbc['marginal']['verdict'], 'marginal_signed_error': sbc['marginal']['signed_error'],
                               'joint': sbc['joint']['verdict'], 'joint_signed_error': sbc['joint']['signed_error'],
                               'per_dim': [p['verdict'] for p in sbc['marginal']['per_dim']]}
    o = np.stack(oracle)
    out['oracle_coverage'] = {str(d): float(coverage(o, gt, delta=d).mean()) for d in DELTAS}
    if split == 'test':
        s = evaluate(oracle_sampler(task, y), benchmark, 'oracle', verbose=False)
        out['oracle_row'] = {k: s[k] for k in ('err_post', 'pos_err_mm', 'success_rate', 'coverage') if k in s}
    print(json.dumps(out, indent=2))


def refine(benchmark, n_poses):
    task = benchmark[4:]
    y, x_true = load_test_conditions(benchmark, 'val')
    _, _, base = load(cache_path(task, y, N_GT, GT_SEED))
    _, fine = generate(task, y[:n_poses], n_gt=10, seed=1, n_seeds=2 * N_SEEDS, h_max=H_MAX / 2,
                       x_true=x_true[:n_poses], n_draws=1, verbose=False)
    rel = np.abs(fine['loop_mass'] - base['loop_mass'][:n_poses]) / base['loop_mass'][:n_poses]
    print(json.dumps({'benchmark': benchmark, 'n_poses': n_poses,
                      'same_loop_count': bool((fine['n_loops'] == base['n_loops'][:n_poses]).all()),
                      'max_rel_mass_change': float(rel.max()), 'median_rel_mass_change': float(np.median(rel))},
                     indent=2))


class Direct7(FiberProblem):
    """6-D fiber in all 7 joints: [p - p*, rotation error] with the geometric Jacobian."""
    dim = 7

    def __init__(self, y_star, device):
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float64), device=device)
        self.device, self.n_targets = device, len(y)
        self.p, self.R = y[:, :3], K.decode_rotation(y)
        lo, hi = K.joint_limits()
        self.lower, self.upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def residual(self, th, idx):
        J, p, R = K.jacobian(th)
        E = R @ self.R[idx].transpose(1, 2)
        e = 0.5 * torch.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], 1)
        return torch.cat([p - self.p[idx], e], 1), J

    def discovery_residual(self, th, idx):
        J, p, R = K.jacobian(th)
        dR = torch.stack([torch.cross(J[:, 3:, j:j + 1].expand(-1, -1, 3), R, dim=1) for j in range(7)], dim=3)
        r = torch.cat([p - self.p[idx], (R - self.R[idx]).reshape(len(th), 9)], 1)
        return r, torch.cat([J[:, :3], dR.reshape(len(th), 9, 7)], 1)

    def weight(self, th, idx):
        return (into_box(th, self.lower) <= self.upper).all(1).double()


def permutation_pvalue(a, b, rng, n_perm=200):
    z = np.concatenate([a, b])
    Kz = imq_kernel(sq_dists(z, z))
    np.fill_diagonal(Kz, 0.0)
    total = Kz.sum()

    def stat(v):
        na, nb = v.sum(), len(v) - v.sum()
        Kv = Kz @ v
        s_aa, s_ab = v @ Kv, (1 - v) @ Kv
        return s_aa / (na * (na - 1)) + (total - s_aa - 2 * s_ab) / (nb * (nb - 1)) - 2 * s_ab / (na * nb)

    v = (np.arange(len(z)) < len(a)).astype(np.float64)
    obs = stat(v)
    return (1 + sum(stat(rng.permutation(v)) >= obs for _ in range(n_perm))) / (1 + n_perm)


def direct(n_poses):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y, _ = load_test_conditions('fr3_6d', 'val')
    gt, _, _ = load(cache_path('6d', y, N_GT, GT_SEED))
    prob = Direct7(y[:n_poses], device)
    gen = torch.Generator().manual_seed(2)
    loops, diag = discover_and_trace(prob, N_SEEDS, gen, h_max=H_MAX)
    rng = np.random.RandomState(0)
    pvals = []
    for i in range(n_poses):
        th, _ = sample_loops(prob, loops[i], i, N_GT, gen)
        pvals.append(permutation_pvalue(into_box(th, prob.lower).cpu().numpy(), gt[i], rng))
    print(json.dumps({'n_poses': n_poses, 'loops_per_pose': diag['n_loops'].tolist(),
                      'all_closed': bool(diag['all_closed'].all()),
                      'pvalues': [round(p, 3) for p in pvals], 'min_p': min(pvals)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('what', choices=['report', 'refine', 'direct'])
    p.add_argument('benchmark', nargs='?', default='fr3_6d', choices=['fr3_6d', 'fr3_5d'])
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--n-poses', type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.what == 'report':
        report(args.benchmark, args.split)
    elif args.what == 'refine':
        refine(args.benchmark, args.n_poses)
    else:
        direct(args.n_poses)


if __name__ == '__main__':
    main()
