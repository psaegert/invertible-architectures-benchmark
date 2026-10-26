# SPDX-License-Identifier: MIT
"""Our FR3 forward kinematics against Jrl's independent FR3 model (fixture
written by robot_ik/scripts/crosscheck_jrl.py; Jrl is not a dependency)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_kinematics')

from robot_ik import kinematics as K  # noqa: E402

FIXTURE = Path(__file__).parent / 'fixtures' / 'jrl_fr3_fk.npz'


def test_flange_pose_matches_jrl():
    with np.load(FIXTURE) as f:
        q, T_hand = f['q'], f['T_hand']
    p, R = K.fk(torch.as_tensor(q))
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # flange -> fr3_hand
    # Jrl computes in float32, hence the tolerances.
    np.testing.assert_allclose(p.numpy(), T_hand[:, :3, 3], atol=2e-6)
    np.testing.assert_allclose(R.numpy() @ Rz, T_hand[:, :3, :3], atol=2e-6)
