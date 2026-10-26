# SPDX-License-Identifier: MIT
"""FR3 tasks in the shared harness: registration, conditions, evaluation end to end
(with a stand-in ground truth; the real one is tested in test_robot_ik_manifold.py)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_kinematics')

import robot_ik.ground_truth as rgt  # noqa: E402
from data.toy_data import MODELS, make_dataset  # noqa: E402
from metrics.benchmarks import forward_fn, load_test_conditions  # noqa: E402
from metrics.evaluate import evaluate, prior_sampler  # noqa: E402


@pytest.mark.parametrize('task, dim_y', [('fr3_6d', 9), ('fr3_5d', 6)])
def test_registered_and_dataset_shapes(task, dim_y):
    assert MODELS[task].n_parameters == 7 and MODELS[task].n_observations == dim_y
    x, y = make_dataset(task, 100, seed=0)
    assert x.shape == (100, 7) and y.shape == (100, dim_y) and x.dtype == np.float32


@pytest.mark.parametrize('task', ['fr3_6d', 'fr3_5d'])
def test_committed_conditions_reproduce(task):
    for split, n in (('test', 1000), ('val', 200)):
        y, x = load_test_conditions(task, split)
        assert y.shape[0] == x.shape[0] == n
        np.testing.assert_allclose(forward_fn(task)(x), y, atol=1e-12)


@pytest.fixture
def stand_in_ground_truth(monkeypatch):
    """Prior draws as 'ground truth': right shapes, wrong distribution."""
    def fake(task, y_star, n_gt=1000, seed=0, verbose=True):
        rng = np.random.RandomState(seed)
        return [MODELS['fr3_6d']().sample_prior(n_gt, rng) for _ in y_star]
    monkeypatch.setattr(rgt, 'posterior_samples', fake)


@pytest.mark.parametrize('task', ['fr3_6d', 'fr3_5d'])
def test_evaluate_end_to_end(task, tmp_path, stand_in_ground_truth):
    _, x_true = load_test_conditions(task)

    def oracle(y, n):   # the generating configuration: zero pose error
        return np.repeat(x_true[:len(y), None], n, axis=1)

    s = evaluate(oracle, task, 'xtrue', n_samples=20, n_conditions=8, out_dir=tmp_path, verbose=False)
    ang = 'rot_err_deg' if task == 'fr3_6d' else 'axis_err_deg'
    assert s['pos_err_mm']['p95'] < 1e-6 and s[ang]['p95'] < 1e-5
    assert s['success_rate']['1mm_1deg'] == 1.0 and s['joint_limit_violation_fraction'] == 0.0

    b = evaluate(prior_sampler(task), task, 'prior', n_samples=20, n_conditions=8, out_dir=tmp_path, verbose=False)
    assert b['pos_err_mm']['median'] > 100 and b['success_rate']['1mm_1deg'] == 0.0

    saved = json.loads((tmp_path / f'{task}_xtrue.json').read_text())
    assert saved['pos_err_mm'] == s['pos_err_mm']
    with np.load(tmp_path / f'{task}_xtrue.npz') as f:
        assert f['coverage'].shape == (8,) and f['pos_err_mm_median'].shape == (8,)


def test_validation_split(tmp_path, stand_in_ground_truth):
    _, x_val = load_test_conditions('fr3_6d', 'val')
    s = evaluate(lambda y, n: np.repeat(x_val[:len(y), None], n, axis=1), 'fr3_6d', 'xtrue_val',
                 n_samples=10, n_conditions=4, out_dir=tmp_path, verbose=False, split='val')
    assert s['split'] == 'val' and s['pos_err_mm']['p95'] < 1e-6
