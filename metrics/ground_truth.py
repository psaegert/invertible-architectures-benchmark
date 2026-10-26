"""Ground-truth posteriors p(x | y*) by rejection sampling, cached to disk.

Both forward processes are deterministic, y = f(x). To get a proper likelihood
we treat the observation as y | x ~ N(f(x), sigma^2 I) with a small fixed sigma
(GT_SIGMA, per benchmark), and sample

    x ~ prior,   accept with probability  exp(-||f(x) - y*||^2 / (2 sigma^2)).

This is exact rejection sampling from p(x | y*) under that likelihood (the
acceptance probability is the likelihood divided by its maximum, 1).

Choice of sigma (identical for every y* and every model):
  kinematics  sigma = 0.01. y* spans roughly [-1.3, 2.0] x [-2.2, 2.2]. The
      ground-truth samples themselves then have a re-simulation error of
      2 sigma^2 = 2e-4, well below the best Err_resim in Table 1 (0.008), and
      the induced blur in x is about 0.01, below the smallest MMD bandwidth
      (0.05), so the kernel cannot resolve it.
  ballistics  sigma = 0.02. y* spans [0, 21]. The upstream forward process
      reports the impact at the last point of a 1500-step time grid, so f is
      piecewise constant with steps of roughly 0.01-0.03 in y. A much smaller
      sigma would only resolve that discretization, not the physics. Ground
      truth re-simulation error sigma^2 = 4e-4 vs. 0.019 best in Table 2.

The proposals are shared between the y*: each chunk of prior samples is pushed
through f once and matched against all pending y* with a KD-tree over f(x)
(neighbors within 4 sigma; exp(-8) ~ 3e-4 of the likelihood mass is cut off).
Within one y* the accepted samples are i.i.d. from the posterior; across y*
they may share proposals, which does not bias the per-y* estimate.

A y* far in the tails of p(y) may not reach N_GT accepted samples before
max_proposals. Its posterior is then estimated from fewer samples (counted in
the result; MMD with the unbiased estimator handles unequal set sizes).

Prior-conditioned ground truth (docs/conditional_fmpe.md): `prior_params`
replaces the prior's parameters (data.toy_data.PRIOR_PARAMS: ballistics v0 ~
Poisson(lambda), kinematics joint sigmas 1..3), so the reference is
p(x | y*, prior_params). None or the benchmark values give the benchmark
posterior and share its cache file.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from data.toy_data import MODELS, PRIOR_PARAMS, prior_tag
from metrics.benchmarks import forward_fn, load_test_conditions, prior

N_GT = 1000
GT_SEED = 777
GT_SIGMA = {'kinematics': 0.01, 'ballistics': 0.02}
GT_CHUNK = {'kinematics': 2_000_000, 'ballistics': 200_000}
GT_MAX_PROPOSALS = {'kinematics': 2_000_000_000, 'ballistics': 50_000_000}
CACHE_DIR = Path(__file__).parent / 'cache'


def rejection_sample(benchmark, y_star, n_gt=N_GT, sigma=None, seed=GT_SEED,
                     chunk=None, max_proposals=None, verbose=True, prior_params=None):
    """Returns (samples, n_proposals): samples is a list with one (n_i, 4) array
    per y*, n_i <= n_gt. prior_params: see the module docstring (None = benchmark prior)."""
    sigma = GT_SIGMA[benchmark] if sigma is None else sigma
    chunk = GT_CHUNK[benchmark] if chunk is None else chunk
    max_proposals = GT_MAX_PROPOSALS[benchmark] if max_proposals is None else max_proposals
    y_star = np.asarray(y_star, dtype=np.float64)
    rng = np.random.RandomState(seed)
    f = forward_fn(benchmark)

    accepted = [[] for _ in y_star]
    counts = np.zeros(len(y_star), dtype=int)
    n_proposals, t0 = 0, time.time()
    while n_proposals < max_proposals:
        pending = np.flatnonzero(counts < n_gt)
        if len(pending) == 0:
            break
        batch = min(chunk, max_proposals - n_proposals)
        x = prior(benchmark, batch, rng, prior_params)
        y = f(x)
        u = rng.rand(batch)
        n_proposals += batch
        ok = np.isfinite(y).all(1)  # ballistics draws without an impact can never match
        x, y, u = x[ok], y[ok], u[ok]

        tree = cKDTree(y)
        for i, idx in zip(pending, tree.query_ball_point(y_star[pending], r=4 * sigma)):
            if not idx:
                continue
            idx = np.asarray(idx)
            d2 = ((y[idx] - y_star[i]) ** 2).sum(1)
            hit = idx[u[idx] < np.exp(-d2 / (2 * sigma ** 2))]
            hit = hit[:n_gt - counts[i]]
            accepted[i].append(x[hit])
            counts[i] += len(hit)

        if verbose:
            print(f'\r  {benchmark}: {n_proposals:.2e} proposals, '
                  f'{(counts >= n_gt).sum()}/{len(y_star)} y* complete, '
                  f'min {counts.min()} samples, {time.time() - t0:.0f}s', end='', flush=True)
    if verbose:
        print()
    samples = [np.concatenate(a) if a else np.empty((0, 4)) for a in accepted]
    return samples, n_proposals


def _cache_path(benchmark, y_star, n_gt, sigma, seed, max_proposals, prior_params=None):
    key, tag = repr((n_gt, sigma, seed, max_proposals)), ''
    if prior_params is not None:
        p = tuple(float(v) for v in np.ravel(prior_params))
        key += repr(p[0]) if len(p) == 1 else repr(p)
        tag = '_' + prior_tag(benchmark, p)
    key = hashlib.sha1(np.ascontiguousarray(y_star, dtype=np.float64).tobytes() + key.encode()).hexdigest()[:10]
    return CACHE_DIR / f'gt_{benchmark}{tag}_sigma{sigma}_n{n_gt}_{key}.npz'


def ground_truth_posterior(benchmark, y_star=None, n_gt=N_GT, sigma=None, seed=GT_SEED,
                           max_proposals=None, verbose=True, prior_params=None):
    """Cached ground-truth posterior samples for each y* (default: the shared
    test conditions). Returns a list of (n_i, 4) arrays, one per y*.
    prior_params: the prior's parameters (module docstring); None = benchmark prior.

    The cache key covers y*, n_gt, sigma, seed, max_proposals and prior_params, so
    changing any of them recomputes instead of silently reusing stale samples."""
    if y_star is None:
        y_star, _ = load_test_conditions(benchmark)
    problem = MODELS[benchmark]()
    if hasattr(problem, 'ground_truth'):  # FR3: exact posterior from robot_ik, not rejection sampling
        if prior_params is not None:
            raise ValueError(f'{benchmark} has no prior parameters')
        return problem.ground_truth(y_star, n_gt=n_gt, seed=seed, verbose=verbose)
    if prior_params is not None and np.array_equal(np.ravel(prior_params), PRIOR_PARAMS[benchmark]['default']):
        prior_params = None  # same draws as the benchmark prior (tests/test_conditional_fmpe.py)
    sigma = GT_SIGMA[benchmark] if sigma is None else sigma
    max_proposals = GT_MAX_PROPOSALS[benchmark] if max_proposals is None else max_proposals
    path = _cache_path(benchmark, y_star, n_gt, sigma, seed, max_proposals, prior_params)

    if not path.exists():
        _compute_and_cache(path, benchmark, y_star, n_gt, sigma, seed, max_proposals, verbose, prior_params)
    with np.load(path) as f:
        return np.split(f['samples'].astype(np.float64), np.cumsum(f['counts'])[:-1])


def _compute_and_cache(path, benchmark, y_star, n_gt, sigma, seed, max_proposals, verbose, prior_params=None):
    samples, n_proposals = rejection_sample(benchmark, y_star, n_gt, sigma, seed,
                                            max_proposals=max_proposals, verbose=verbose, prior_params=prior_params)
    counts = np.array([len(s) for s in samples])
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(path, samples=np.concatenate(samples).astype(np.float32), counts=counts,
             y_star=y_star, sigma=sigma, n_proposals=n_proposals)
    if verbose:
        short = counts < n_gt
        print(f'  saved {path.name}; {short.sum()} y* below {n_gt} samples'
              + (f' (min {counts.min()})' if short.any() else ''))


if __name__ == '__main__':
    import sys
    # python -m metrics.ground_truth ballistics:12 kinematics:0.35,0.35,0.35
    # (ground truth under v0 ~ Poisson(12), resp. joint sigmas 0.35)
    for b in sys.argv[1:] or ['kinematics', 'ballistics']:
        b, _, p = b.partition(':')
        ground_truth_posterior(b, prior_params=[float(v) for v in p.split(',')] if p else None)
