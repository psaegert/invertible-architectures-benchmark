# SPDX-License-Identifier: MIT
"""FR3 kinematics: URDF values, Jacobians against finite differences, encodings."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_kinematics')

from robot_ik import kinematics as K  # noqa: E402


@pytest.fixture
def q():
    return torch.as_tensor(K.sample_uniform(16, np.random.RandomState(0)))


def test_limits_match_franka_yaml():
    lo, hi = K.joint_limits()
    np.testing.assert_allclose(lo, [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
    np.testing.assert_allclose(hi, [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])


def test_zero_configuration():
    p, R = K.fk(torch.zeros(1, 7, dtype=torch.float64))
    np.testing.assert_allclose(p[0].numpy(), [0.088, 0.0, 0.926], atol=1e-12)
    np.testing.assert_allclose(R[0].numpy(), np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_samples_inside_limits():
    lo, hi = K.joint_limits()
    x = K.sample_uniform(10_000, np.random.RandomState(1))
    assert ((x >= lo) & (x <= hi)).all()


def test_jacobian_matches_finite_differences(q):
    J, p, R = K.jacobian(q)
    h = 1e-6
    for j in range(7):
        dq = torch.zeros_like(q)
        dq[:, j] = h
        pp, Rp = K.fk(q + dq)
        pm, Rm = K.fk(q - dq)
        v = (pp - pm) / (2 * h)
        W = (Rp - Rm) / (2 * h) @ R.transpose(1, 2)   # skew(omega)
        w = torch.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], dim=1)
        np.testing.assert_allclose(J[:, :3, j].numpy(), v.numpy(), atol=1e-8)
        np.testing.assert_allclose(J[:, 3:, j].numpy(), w.numpy(), atol=1e-8)


def test_flange_position_and_axis_ignore_joint7(q):
    p, R = K.fk(q)
    q2 = q.clone()
    q2[:, 6] = -q2[:, 6] + 0.3
    p2, R2 = K.fk(q2)
    np.testing.assert_allclose(p2.numpy(), p.numpy(), atol=1e-12)
    np.testing.assert_allclose(R2[:, :, 2].numpy(), R[:, :, 2].numpy(), atol=1e-12)


def test_encode_decode_round_trip(q):
    p, R = K.fk(q)
    y = K.encode(p, R, '6d')
    assert y.shape == (16, 9) and K.encode(p, R, '5d').shape == (16, 6)
    np.testing.assert_allclose(K.decode_rotation(y).numpy(), R.numpy(), atol=1e-12)


def test_tangent_basis_is_orthonormal(q):
    _, R = K.fk(q)
    a = R[:, :, 2]
    u1, u2 = K.tangent_basis(a)
    G = torch.stack([a, u1, u2], dim=2)
    np.testing.assert_allclose((G.transpose(1, 2) @ G).numpy(), np.broadcast_to(np.eye(3), G.shape), atol=1e-12)


def test_pa_jacobian_matches_finite_differences(q):
    th = q[:, :6]
    _, R = K.fk(q)
    u1, u2 = K.tangent_basis(R[:, :, 2])
    J5, p, a = K.pa_jacobian(th, u1, u2)
    h = 1e-6

    def chart(t):
        pq, Rq = K.fk(torch.cat([t, torch.zeros_like(t[:, :1])], dim=1))
        aq = Rq[:, :, 2]
        return torch.cat([pq, (u1 * aq).sum(1, keepdim=True), (u2 * aq).sum(1, keepdim=True)], dim=1)

    for j in range(6):
        dt = torch.zeros_like(th)
        dt[:, j] = h
        fd = (chart(th + dt) - chart(th - dt)) / (2 * h)
        np.testing.assert_allclose(J5[:, :, j].numpy(), fd.numpy(), atol=1e-8)


def test_roll_of_recovers_joint7(q):
    _, R = K.fk(q)
    np.testing.assert_allclose(K.roll_of(q[:, :6], R).numpy(), q[:, 6].numpy(), atol=1e-12)
