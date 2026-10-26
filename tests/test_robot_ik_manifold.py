# SPDX-License-Identifier: MIT
"""The fiber tracer on problems with a known answer.

Ellipse x^2/a^2 + y^2/b^2 = 1 (Euclidean): the coarea posterior of a uniform
prior is uniform in the parameter phi of (a cos phi, b sin phi), and its
tau-mass is pi a b (ds / |grad F| = (a b / 2) d phi).

Planar 3R arm (torus, position target): the traced posterior against
rejection sampling with a narrow Gaussian likelihood, for a box prior and a
Gaussian prior; the gap must be within the gap between two rejection runs.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip('torch')

from metrics.mmd import imq_kernel, sq_dists  # noqa: E402
from robot_ik.manifold import FiberProblem, discover_and_trace, sample_loops, trace  # noqa: E402


class Ellipse(FiberProblem):
    """F(x, y) = x^2 / a^2 + y^2 / b^2 - 1 (one target); uniform prior."""
    periodic = False
    dim, n_targets, device = 2, 1, 'cpu'

    def __init__(self, a, b):
        self.a, self.b = a, b

    def residual(self, th, idx):
        x, y = th[:, 0], th[:, 1]
        r = (x ** 2 / self.a ** 2 + y ** 2 / self.b ** 2 - 1)[:, None]
        J = torch.stack([2 * x / self.a ** 2, 2 * y / self.b ** 2], dim=1)[:, None, :]
        return r, J

    discovery_residual = residual

    def weight(self, th, idx):
        return torch.ones(len(th), dtype=th.dtype)


def test_ellipse_mass_and_uniform_phase():
    a, b = 1.3, 0.5
    prob = Ellipse(a, b)
    gen = torch.Generator().manual_seed(0)
    start = torch.tensor([[a, 0.0]], dtype=torch.float64)
    (loop,) = trace(prob, start, torch.zeros(1, dtype=torch.long), h_max=0.01)
    assert loop.closed and loop.closure_error < 1e-8
    # |c| = |grad F| = 2 sqrt(x^2/a^4 + y^2/b^4) and ds / |c| = (a b / 2) d phi, so the tau-length is pi a b.
    assert abs(loop.mass - math.pi * a * b) / (math.pi * a * b) < 1e-5
    x, _ = sample_loops(prob, [loop], 0, 20_000, gen)
    phi = np.arctan2(x[:, 1].numpy() / b, x[:, 0].numpy() / a)
    assert stats.kstest((phi + np.pi) / (2 * np.pi), 'uniform').pvalue > 0.01
    r, _ = prob.residual(x, None)
    assert r.abs().max() < 1e-12


class TwoCircles(FiberProblem):
    """F = f1 f2, f1 = x^2 + y^2 - 1, f2 = (x - 3)^2 + y^2 - 0.25: two disjoint loops."""
    periodic = False
    dim, n_targets, device = 2, 1, 'cpu'

    def residual(self, th, idx):
        x, y = th[:, 0], th[:, 1]
        f1, f2 = x ** 2 + y ** 2 - 1, (x - 3) ** 2 + y ** 2 - 0.25
        g = torch.stack([2 * x * f2 + 2 * (x - 3) * f1, 2 * y * f2 + 2 * y * f1], dim=1)
        return (f1 * f2)[:, None], g[:, None, :]

    discovery_residual = residual

    def weight(self, th, idx):
        return torch.ones(len(th), dtype=th.dtype)


def test_draws_split_between_loops_by_mass():
    """With several loops, each gets its share of the posterior mass."""
    prob = TwoCircles()
    gen = torch.Generator().manual_seed(4)
    starts = torch.tensor([[1.0, 0.0], [3.5, 0.0]], dtype=torch.float64)
    loops = trace(prob, starts, torch.zeros(2, dtype=torch.long), h_max=0.01)
    assert all(lp.closed for lp in loops)
    share = loops[0].mass / (loops[0].mass + loops[1].mass)
    x, _ = sample_loops(prob, loops, 0, 4000, gen)
    on_first = (x[:, 0] < 2).double().mean().item()
    assert abs(on_first - share) < 4 * math.sqrt(share * (1 - share) / 4000), (on_first, share)


LENGTHS = (1.0, 0.8, 0.6)


def planar_fk(th):
    """End point of a planar 3R arm, (n, 2); works for numpy and torch."""
    lib = torch if isinstance(th, torch.Tensor) else np
    c = lib.cumsum(th, 1) if lib is np else torch.cumsum(th, 1)
    return lib.stack([sum(l * lib.cos(c[:, k]) for k, l in enumerate(LENGTHS)),
                      sum(l * lib.sin(c[:, k]) for k, l in enumerate(LENGTHS))], 1)


class Planar3R(FiberProblem):
    dim, device = 3, 'cpu'

    def __init__(self, targets, lower, upper, prior_sigma=None):
        self.t = torch.as_tensor(targets, dtype=torch.float64)
        self.n_targets = len(self.t)
        self.lower, self.upper, self.sigma = lower, upper, prior_sigma

    def residual(self, th, idx):
        c = torch.cumsum(th, 1)
        p = planar_fk(th)
        J = torch.zeros(len(th), 2, 3, dtype=th.dtype)
        for j in range(3):
            J[:, 0, j] = -sum(l * torch.sin(c[:, k]) for k, l in enumerate(LENGTHS) if k >= j)
            J[:, 1, j] = sum(l * torch.cos(c[:, k]) for k, l in enumerate(LENGTHS) if k >= j)
        return p - self.t[idx], J

    discovery_residual = residual

    def weight(self, th, idx):
        w = torch.remainder(th + math.pi, 2 * math.pi) - math.pi        # principal angles
        box = ((w >= self.lower) & (w <= self.upper)).all(1).double()
        if self.sigma is None:
            return box
        return box * torch.exp(-0.5 * (w ** 2).sum(1) / self.sigma ** 2)


def rejection(target, lower, upper, prior_sigma, n, rng, sigma_y=0.004, batch=4_000_000):
    """Posterior under a Gaussian likelihood of width sigma_y, by rejection."""
    out = []
    while sum(len(o) for o in out) < n:
        if prior_sigma is None:
            th = rng.uniform(lower, upper, size=(batch, 3))
        else:
            th = rng.normal(0, prior_sigma, size=(batch, 3))
            th = th[((th >= lower) & (th <= upper)).all(1)]
        d2 = ((planar_fk(th) - target) ** 2).sum(1)
        out.append(th[rng.rand(len(th)) < np.exp(-0.5 * d2 / sigma_y ** 2)])
    return np.concatenate(out)[:n]


# name: (target, lower, upper, prior sigma or None for a uniform box prior)
PLANAR_CASES = {'box': ((1.1, 0.7), -2.5, 2.5, None), 'gauss': ((1.1, 0.7), -2.5, 2.5, 1.0)}
FIXTURE = Path(__file__).parent / 'fixtures' / 'planar3r_rejection.npz'


@pytest.mark.parametrize('case', list(PLANAR_CASES))
def test_planar_3r_matches_rejection(case):
    """The traced posterior is as close to rejection sampling as two rejection
    runs are to each other, and clearly closer than the same curve sampled
    uniformly in arclength (which drops the coarea weight 1 / |c|)."""
    target, lower, upper, prior_sigma = PLANAR_CASES[case]
    prob = Planar3R(np.asarray(target)[None], lower, upper, prior_sigma)
    gen = torch.Generator().manual_seed(1)
    loops, diag = discover_and_trace(prob, 500, gen, h_max=0.02)
    assert diag['all_closed'].all() and diag['max_closure_error'][0] < 1e-8
    x, _ = sample_loops(prob, loops[0], 0, 2000, gen)
    principal = lambda t: (torch.remainder(t + math.pi, 2 * math.pi) - math.pi).numpy()  # noqa: E731
    # Wrong on purpose: resample tau-uniform draws with weight |c|, i.e. uniform in arclength.
    many, _ = sample_loops(prob, loops[0], 0, 20_000, gen)
    _, J = prob.residual(many, torch.zeros(len(many), dtype=torch.long))
    w = torch.linalg.det(J @ J.transpose(1, 2)).sqrt()
    wrong = many[torch.multinomial(w, 2000, replacement=True, generator=gen)]
    with np.load(FIXTURE) as f:
        ref = np.concatenate([f[f'{case}_1'], f[f'{case}_2']])
    rng = np.random.RandomState(0)
    assert mmd_permutation_pvalue(principal(x), ref, rng) > 0.01
    assert mmd_permutation_pvalue(principal(wrong), ref, rng) < 0.005


def mmd_permutation_pvalue(a, b, rng, n_perm=200):
    """Two-sample permutation test with the repo's MMD kernel (metrics/mmd.py)."""
    z = np.concatenate([a, b])
    K = imq_kernel(sq_dists(z, z))
    np.fill_diagonal(K, 0.0)

    total = K.sum()

    def stat(mask):
        v = mask.astype(np.float64)
        na, nb = v.sum(), len(v) - v.sum()
        Kv = K @ v
        s_aa, s_ab = v @ Kv, (1 - v) @ Kv
        s_bb = total - s_aa - 2 * s_ab
        return s_aa / (na * (na - 1)) + s_bb / (nb * (nb - 1)) - 2 * s_ab / (na * nb)

    mask = np.arange(len(z)) < len(a)
    observed = stat(mask)
    perms = [stat(rng.permutation(mask)) for _ in range(n_perm)]
    return (1 + sum(p >= observed for p in perms)) / (1 + n_perm)


@pytest.mark.parametrize('task', ['6d', '5d'])
def test_fr3_draws_solve_the_pose_inside_the_limits(task):
    """Ground-truth draws for one FR3 pose reproduce y* to numerical precision, stay
    inside the joint limits, and the generating configuration lies on a traced loop."""
    pytest.importorskip('pytorch_kinematics')
    from metrics.benchmarks import load_test_conditions
    from robot_ik.ground_truth import generate
    from robot_ik.metrics import joint_limit_violations, pose_errors
    y, x_true = load_test_conditions(f'fr3_{task}', 'val')
    draws, diag = generate(task, y[:1], n_gt=200, seed=3, n_seeds=150, h_max=0.05, device='cpu', verbose=False,
                           x_true=x_true[:1], n_draws=1)
    x = np.stack(draws[0])
    pos, ang = pose_errors(task, x, y[:1])
    assert pos.max() < 1e-6 and ang.max() < 1e-6          # mm, deg
    assert not joint_limit_violations(x).any()
    assert diag['all_closed'].all() and diag['x_true_on_fiber'].all()
