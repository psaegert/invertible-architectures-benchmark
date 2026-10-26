# SPDX-License-Identifier: MIT
"""Exact ground-truth posteriors p(x | y*) for the FR3 tasks, cached to disk.

The flange position p and z axis a depend on joints 1-6 only, so for both tasks
the solutions of a target live on the fiber C = {theta_1..6 : p = p*, a = a*}:
five constraints in six joints, a curve. robot_ik/manifold.py traces it and
samples it exactly (coarea weight, uniform prior within the joint limits).
    5-D task  theta_1..6 from C inside the box of joints 1-6, joint 7 uniform
              within its limits (roll about a is free).
    6-D task  joint 7 is fixed on C by the remaining roll; keep the parts of C
              where joints 1-6 and that joint 7 are inside the limits. The
              6-D coarea weight equals the 5-D one along C, so the same
              tau-uniform sampling is exact.

Generation is expensive, so the sets for the committed conditions are made once
on a GPU (robot_ik/scripts/generate_gt.py) and loaded from metrics/cache. Each
cache file also holds a second, independent draw (the "oracle" model: the noise
floor of every distributional metric) and per-pose diagnostics.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from robot_ik import kinematics as K
from robot_ik.manifold import FiberProblem, discover_and_trace, into_box, sample_loops

CACHE_DIR = Path(__file__).resolve().parents[1] / 'metrics' / 'cache'
N_SEEDS = 10_000   # discovery seeds per pose
H_MAX = 0.02       # largest tracing step (rad of joint-space arclength)


class Fr3Fibers(FiberProblem):
    """The curves C of a batch of FR3 targets, in joints 1-6."""
    dim = 6

    def __init__(self, task, y_star, device='cpu'):
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float64), device=device)
        self.task, self.device, self.n_targets = task, device, len(y)
        self.p = y[:, :3]
        if task == '6d':
            self.R = K.decode_rotation(y)
            self.a = self.R[:, :, 2]
        else:
            self.R = None
            self.a = torch.nn.functional.normalize(y[:, 3:6], dim=1)
        self.u1, self.u2 = K.tangent_basis(self.a)
        lo, hi = K.joint_limits()
        self.lower = torch.as_tensor(lo, device=device)
        self.upper = torch.as_tensor(hi, device=device)

    def residual(self, th, idx):
        J, p, a = K.pa_jacobian(th, self.u1[idx], self.u2[idx])
        r = torch.cat([p - self.p[idx], (self.u1[idx] * a).sum(1, keepdim=True),
                       (self.u2[idx] * a).sum(1, keepdim=True)], dim=1)
        return r, J

    def discovery_residual(self, th, idx):
        """[p - p*, a - a*], valid everywhere (the chart residual also vanishes at a = -a*)."""
        J, p, R = K.jacobian(torch.cat([th, torch.zeros_like(th[:, :1])], dim=1))
        a = R[:, :, 2]
        da = torch.cross(J[:, 3:, :6], a[:, :, None].expand(-1, -1, 6), dim=1)
        return torch.cat([p - self.p[idx], a - self.a[idx]], dim=1), torch.cat([J[:, :3, :6], da], dim=1)

    def joint7(self, th, idx):
        """6-D task: joint 7 fixed by the target's roll, as a representative in its box."""
        q7 = K.roll_of(th, self.R[idx])
        return into_box(q7, self.lower[6])

    def weight(self, th, idx):
        inside = (into_box(th, self.lower[:6]) <= self.upper[:6]).all(1)
        if self.task == '6d':
            inside = inside & (self.joint7(th, idx) <= self.upper[6])
        return inside.double()


def generate(task, y_star, n_gt=1000, seed=0, verbose=True, n_seeds=N_SEEDS, h_max=H_MAX, device=None,
             n_draws=2, x_true=None):
    """Posterior draws for every y*. Returns (draws, diagnostics): draws is a list
    of n_draws lists of (n_gt, 7) arrays (the first is the ground truth, the
    second the independent oracle draw).

    x_true (optional, the configurations that generated y*) is used for one
    diagnostic only, never as a seed: whether it lies on a traced loop. Over many
    poses, the share that does not is an unbiased estimate of missed mass."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator().manual_seed(seed)
    prob = Fr3Fibers(task, y_star, device)
    t0 = time.time()
    loops, diag = discover_and_trace(prob, n_seeds, gen, h_max=h_max, verbose=verbose)
    if verbose:
        print(f'  traced {int(diag["n_loops"].sum())} loops for {len(loops)} poses in {time.time() - t0:.0f}s', flush=True)
    draws = [[] for _ in range(n_draws)]
    accepted = np.zeros(len(loops))
    lo7, hi7 = float(prob.lower[6]), float(prob.upper[6])
    for i, lp in enumerate(loops):
        if not lp:
            raise RuntimeError(f'pose {i}: no solution curve found')
        for d in range(n_draws):
            th, acc = sample_loops(prob, lp, i, n_gt, gen)
            if len(th) < n_gt:
                raise RuntimeError(f'pose {i}: only {len(th)} in-limit draws (in-box share {acc:.2e})')
            ii = torch.full((n_gt,), i, device=device)
            if task == '6d':
                q7 = prob.joint7(th, ii)
            else:
                q7 = lo7 + (hi7 - lo7) * torch.rand(n_gt, generator=gen, dtype=torch.float64).to(device)
            x = torch.cat([into_box(th, prob.lower[:6]), q7[:, None]], dim=1)
            draws[d].append(x.cpu().numpy())
            accepted[i] = acc if d == 0 else accepted[i]
    if x_true is not None:
        from robot_ik.manifold import on_loops
        xt = torch.as_tensor(np.asarray(x_true, dtype=np.float64)[:, :6], device=device)
        diag['x_true_on_fiber'] = np.array([bool(on_loops(prob, xt[i:i + 1], lp, tol=h_max)[0])
                                            for i, lp in enumerate(loops)])
    diag['in_box_share'] = accepted
    diag['in_box_mass'] = diag['loop_mass'] * accepted
    diag['n_seeds'] = np.array(n_seeds)
    diag['h_max'] = np.array(h_max)
    if verbose:
        print(f'  sampled in {time.time() - t0:.0f}s total', flush=True)
    return draws, diag


def cache_key(task, y_star, n_gt, seed):
    h = hashlib.sha1(np.ascontiguousarray(y_star, dtype=np.float64).tobytes()
                     + repr((task, n_gt, seed)).encode()).hexdigest()[:10]
    return f'gt_fr3_{task}_n{n_gt}_s{seed}_{h}'


def cache_path(task, y_star, n_gt, seed):
    return CACHE_DIR / f'{cache_key(task, y_star, n_gt, seed)}.npz'


def save(path, draws, y_star, diagnostics):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, samples=np.stack(draws[0]).astype(np.float32), oracle=np.stack(draws[1]).astype(np.float32),
             y_star=np.asarray(y_star, dtype=np.float64), **{f'diag_{k}': v for k, v in diagnostics.items()})


def load(path):
    """(ground-truth draws, oracle draws, diagnostics); draws as lists of (n_gt, 7) arrays."""
    with np.load(path) as f:
        samples = list(f['samples'].astype(np.float64))
        oracle = list(f['oracle'].astype(np.float64))
        diagnostics = {k[5:]: f[k] for k in f.files if k.startswith('diag_')}
    return samples, oracle, diagnostics


def posterior_samples(task, y_star, n_gt=1000, seed=0, verbose=True):
    """List of (n_gt, 7) exact posterior draws, one per y*. Loads the cache,
    otherwise generates (about 15 min on an A100 for 1000 poses, hours on a
    CPU) and caches."""
    y_star = np.asarray(y_star, dtype=np.float64)
    path = cache_path(task, y_star, n_gt, seed)
    if not path.exists():
        if verbose:
            print(f'  {path.name} is not cached: generating it (docs/robot_ik.md)', flush=True)
        draws, diagnostics = generate(task, y_star, n_gt=n_gt, seed=seed, verbose=verbose)
        save(path, draws, y_star, diagnostics)
    return load(path)[0]


def oracle_sampler(task, y_star_all):
    """A 'model' that returns the second, independent ground-truth draw: scoring it
    gives the noise floor of MMD, coverage and calibration. Only for the y* of a
    cached ground truth (the committed conditions, with the harness's N_GT and GT_SEED)."""
    from metrics.ground_truth import GT_SEED, N_GT
    path = cache_path(task, np.asarray(y_star_all, dtype=np.float64), N_GT, GT_SEED)
    _, oracle, _ = load(path)
    lookup = {np.asarray(y, dtype=np.float32).tobytes(): o for y, o in zip(y_star_all, oracle)}

    def sample(y_star, n):
        return np.stack([lookup[np.asarray(y, dtype=np.float32).tobytes()][:n] for y in y_star])
    return sample
