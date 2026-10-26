"""Err_post (Eq. 10), Err_resim (Eq. 11) and inference time for any model.

A model only has to provide

    sample(y_star, n) -> x,   y_star: (M, dim_y) float32 array,  x: (M, n, dim_x)

(numpy array or torch tensor, raw x units; dim_x = 4 for the toy benchmarks,
7 for the FR3 tasks). Run e.g.

    python -m metrics.evaluate kinematics --prior-baseline

Problems that define `task_metrics` (the FR3 tasks, robot_ik/metrics.py) add
their own numbers to the summary and the per-y* arrays: pose errors, joint
limit violations and coverage.

Reported numbers:
  Err_post   unbiased MMD^2 estimate (metrics/mmd.py) between n model samples and
             the cached ground-truth samples, per y*, then averaged. The paper
             writes "MMD" in Eq. (10) and defines it by the squared expression of
             Eq. (2); we report the value of that expression. The biased
             V-statistic (adds ~0.006 at n = 1000, see mmd.py) is reported
             alongside as err_post_biased, since the paper does not say which
             estimator or sample size it used.
  Err_resim  mean over model samples of ||f(x) - y*||^2 with the true forward
             process f, per y*, then averaged.

Ballistics: f(x) is undefined (NaN) when a sampled x yields no single ground
impact. Those samples are left out of Err_resim and counted separately as
`resim_failed`; a y* where every sample fails gets Err_resim = NaN. The table
reports the plain mean (which the paper's Table 2 says is distorted by extreme
outliers), a mean with per-y* values clamped at CLAMP, and median + quartiles.
The per-y* arrays are saved for the log-scale boxplots of Figs. 4-5.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from data.toy_data import PRIOR_PARAMS
from metrics.benchmarks import BENCHMARKS, forward_fn, get_model, load_test_conditions, prior
from metrics.ground_truth import ground_truth_posterior
from metrics.mmd import mmd

N_SAMPLES = 1000  # model samples per y*, same as the ground-truth set size
RESULTS_DIR = Path(__file__).parent / 'results'

# Per-y* values are clamped at this value before averaging for the "clamped
# mean" column. The paper (Sec. 4.2) clamps but does not give the value. Its
# clamped Table 2 means reach 4.36 (cVAE Err_post) and 3.67 (INN L2+MMD
# Err_resim), so its clamp is at least that; 10 is the smallest round value
# consistent with every entry. (Err_post is bounded by 2 k(0) = 6 for our
# kernel anyway, so in practice this only clamps Err_resim.)
CLAMP = 10.0


def err_post(model_samples, gt_samples, unbiased=True):
    """Per-y* MMD^2 between model samples (M, n, 4) and a list of M ground-truth
    sample arrays. Returns (per_y (M,), mean)."""
    per_y = np.array([mmd(a, b, unbiased=unbiased) for a, b in zip(model_samples, gt_samples)])
    return per_y, float(np.nanmean(per_y))


def err_resim(model_samples, y_star, forward_fn):
    """Per-y* mean squared re-simulation error. Returns (per_y (M,), mean,
    failed (M,) fraction of samples with no defined f(x))."""
    M, n, d = model_samples.shape
    y = forward_fn(model_samples.reshape(M * n, d)).reshape(M, n, -1)
    sq = ((y - y_star[:, None, :]) ** 2).sum(-1)  # NaN where f(x) undefined
    failed = np.isnan(sq).mean(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows -> NaN, reported via `failed`
        per_y = np.nanmean(sq, axis=1)
    return per_y, float(np.nanmean(per_y)), failed


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except ImportError:
        pass


def timed_sampling(sample_fn, y_star, n, n_warmup=3):
    """Calls sample_fn on the full batch of y* and times it, after warm-up calls
    on a small batch (CUDA init, cuDNN autotuning, allocator). Returns (samples,
    seconds). The time is only comparable between models run on the same
    machine, not with the paper's GTX 1080 Ti numbers."""
    y32 = np.asarray(y_star, dtype=np.float32)
    for _ in range(n_warmup):
        sample_fn(y32[:10], n)
    _sync()
    t0 = time.perf_counter()
    x = sample_fn(y32, n)
    _sync()
    return _to_numpy(x), time.perf_counter() - t0


def summarize(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {**dict.fromkeys(('mean', 'mean_clamped', 'median', 'q1', 'q3', 'max'), None),
                'n_finite': 0}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {'mean': float(v.mean()), 'mean_clamped': float(np.minimum(v, CLAMP).mean()),
            'median': float(med), 'q1': float(q1), 'q3': float(q3), 'max': float(v.max()),
            'n_finite': int(len(v))}


def evaluate(sample_fn, benchmark, name, n_samples=N_SAMPLES, out_dir=RESULTS_DIR, verbose=True,
             n_conditions=None, gt_options=None, prior_params=None, split='test'):
    """Score one model on the shared test conditions; saves the per-y* arrays to
    <out_dir>/<benchmark>_<name>.npz and the summary to .json next to it.
    prior_params (data.toy_data.PRIOR_PARAMS): score against the ground truth under
    that prior instead of the benchmark prior; sample_fn must then sample
    p(x | y*, prior_params).
    split: 'test' (default) or 'val' where a validation set exists (the FR3 tasks);
    select models on 'val', report on 'test'."""
    problem = get_model(benchmark)
    y_star, _ = load_test_conditions(benchmark) if split == 'test' else load_test_conditions(benchmark, split)
    if n_conditions is not None:
        if not 1 <= n_conditions <= len(y_star):
            raise ValueError(f'n_conditions must be between 1 and {len(y_star)}')
        y_star = y_star[:n_conditions]
    if n_samples < 2:
        raise ValueError('MMD evaluation requires at least two samples')
    gt_options = gt_options or {}
    gt = ground_truth_posterior(benchmark, y_star, verbose=verbose, prior_params=prior_params, **gt_options)
    if any(len(g) < 2 for g in gt):
        raise ValueError('Ground truth has fewer than two samples for a condition; increase proposal cap')

    x, seconds = timed_sampling(sample_fn, y_star, n_samples)
    assert x.shape == (len(y_star), n_samples, problem.n_parameters), f'sample() returned {x.shape}'
    if not np.isfinite(x).all():
        raise ValueError('Model produced non-finite posterior samples')

    post, _ = err_post(x, gt)
    post_biased, _ = err_post(x, gt, unbiased=False)
    resim, _, failed = err_resim(x, y_star, forward_fn(benchmark))
    task_summary, task_arrays = problem.task_metrics(x, y_star, gt) if hasattr(problem, 'task_metrics') else ({}, {})

    summary = {
        'benchmark': benchmark, 'model': name, 'n_conditions': len(y_star),
        'n_samples': n_samples, 'n_gt_min': int(min(len(g) for g in gt)),
        'err_post': summarize(post), 'err_post_biased': summarize(post_biased), 'err_resim': summarize(resim),
        'resim_failed_fraction': float(failed.mean()),
        'inference_seconds': seconds,
        'inference_ms_per_condition': 1000 * seconds / len(y_star),
        'clamp': CLAMP,
        'evaluation_scope': 'full' if n_conditions is None and n_samples == N_SAMPLES and not gt_options else 'pilot',
        'gt_options': gt_options,
        **({'split': split} if split != 'test' else {}),
        **task_summary,
    }
    if prior_params is not None:
        summary.update(zip(PRIOR_PARAMS[benchmark]['names'], map(float, np.ravel(prior_params))))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'{benchmark}_{name}.npz', err_post=post, err_post_biased=post_biased, err_resim=resim,
             resim_failed=failed, y_star=y_star, **task_arrays)
    (out_dir / f'{benchmark}_{name}.json').write_text(json.dumps(summary, indent=2))
    if verbose:
        print_row(summary)
    return summary


def print_row(s):
    p, r = s['err_post'], s['err_resim']
    if r['n_finite'] == 0:
        print(f"{s['benchmark']} {s['model']}: all re-simulations failed; Err_resim undefined")
        return
    print(f"{s['benchmark']:<11} {s['model']:<16} "
          f"Err_post {p['mean']:.4f}  Err_resim {r['mean']:.4f}  "
          f"time {s['inference_seconds'] * 1000:.0f} ms ({s['n_conditions']} y* x {s['n_samples']})")
    print(f"{'':<28} Err_post with biased MMD^2 estimator: {s['err_post_biased']['mean']:.4f}")
    print(f"{'':<28} clamped@{s['clamp']:g}: post {p['mean_clamped']:.4f}  resim {r['mean_clamped']:.4f}")
    print(f"{'':<28} median [q1, q3]: post {p['median']:.4f} [{p['q1']:.4f}, {p['q3']:.4f}]  "
          f"resim {r['median']:.4f} [{r['q1']:.4f}, {r['q3']:.4f}]")
    if s['resim_failed_fraction'] > 0:
        print(f"{'':<28} f(x) undefined for {100 * s['resim_failed_fraction']:.2f}% of samples")
    if 'pos_err_mm' in s:
        ang = 'rot_err_deg' if 'rot_err_deg' in s else 'axis_err_deg'
        print(f"{'':<28} pose error median/p95: {s['pos_err_mm']['median']:.2f}/{s['pos_err_mm']['p95']:.2f} mm, "
              f"{s[ang]['median']:.2f}/{s[ang]['p95']:.2f} deg; success@1mm,1deg {s['success_rate']['1mm_1deg']:.3f}; "
              f"limits violated {s['joint_limit_violation_fraction']:.3f}; coverage {s['coverage']['mean']:.3f}")


def prior_sampler(benchmark, seed=0):
    """A deliberately bad 'model' that ignores y* and samples the prior. It
    must score clearly worse than any trained model, or the metrics are not
    conditioning on y*."""
    rng = np.random.RandomState(seed)
    dim = get_model(benchmark).n_parameters
    return lambda y, n: prior(benchmark, len(y) * n, rng).reshape(len(y), n, dim)


def boxplot(result_files, out_path, metric='err_resim'):
    """Log-scale boxplot of per-y* values for several models, as in Figs. 4-5."""
    import matplotlib.pyplot as plt
    data, labels = [], []
    for p in map(Path, result_files):
        with np.load(p) as f:
            v = f[metric]
        data.append(v[np.isfinite(v) & (v > 0)])  # unbiased MMD^2 can be <= 0
        labels.append(p.stem)
    fig, ax = plt.subplots(figsize=(1.2 * len(data) + 2, 4))
    ax.boxplot(data, whis=(5, 95), showfliers=True, flierprops={'markersize': 2})
    ax.set_xticks(range(1, len(data) + 1), labels, rotation=30, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('benchmark', choices=BENCHMARKS)
    p.add_argument('--prior-baseline', action='store_true', help='score the prior-only "model"')
    args = p.parse_args()
    if args.prior_baseline:
        evaluate(prior_sampler(args.benchmark), args.benchmark, 'prior')
