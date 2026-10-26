# SPDX-License-Identifier: MIT
"""Forward kinematics, Jacobians and pose encodings of the Franka FR3.

The robot model is Franka's official description (robot_ik/assets/fr3, see
PROVENANCE.md); kinematics are computed with pytorch_kinematics. Units are
metres and radians. The end effector is the flange `fr3_link8`, 0.107 m past
joint 7 along the joint-7 axis.

Joint 7 rotates the flange about its own z axis, so the flange position p and
z axis a depend on joints 1-6 only. `pa_jacobian` gives the Jacobian of (p, a)
in those six joints, written in an orthonormal chart of R^3 x S^2; the ground
truth (robot_ik/manifold.py) is built on it.

Pose encodings (what the models are conditioned on):
    6-D task   y = [p (3), r1 (3), r2 (3)]   first two columns of the flange
               rotation matrix (Zhou et al. 2019); 9 numbers.
    5-D task   y = [p (3), a (3)]            a = third column, the flange z
               axis; roll about a is free; 6 numbers.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

URDF = Path(__file__).parent / 'assets' / 'fr3' / 'fr3.urdf'
N_JOINTS = 7


@lru_cache(maxsize=None)
def _chain(dtype, device):
    import pytorch_kinematics as pk
    return pk.build_serial_chain_from_urdf(URDF.read_text(), 'fr3_link8', 'fr3_link0').to(dtype=dtype, device=device)


def joint_limits():
    """(lower, upper) as float64 numpy arrays of shape (7,), from the URDF."""
    lo, hi = _chain(torch.float64, 'cpu').get_joint_limits()
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def sample_uniform(n, rng=np.random):
    """n joint configurations uniform within the joint limits, (n, 7) float64."""
    lo, hi = joint_limits()
    return lo + (hi - lo) * rng.rand(n, N_JOINTS)


def _as_tensor(q):
    if isinstance(q, torch.Tensor):
        return q
    return torch.as_tensor(np.asarray(q), dtype=torch.float64)


def fk(q):
    """Flange pose for configurations q (N, 7): returns (p (N, 3), R (N, 3, 3)),
    same dtype and device as q (numpy input is treated as float64)."""
    q = _as_tensor(q)
    m = _chain(q.dtype, q.device).forward_kinematics(q).get_matrix()
    return m[:, :3, 3], m[:, :3, :3]


def jacobian(q):
    """Geometric Jacobian in the base frame at the flange origin, (N, 6, 7):
    rows [linear velocity (3); angular velocity (3)]. Also returns (p, R)."""
    q = _as_tensor(q)
    J, T = _chain(q.dtype, q.device).jacobian_tensor(q, ret_eef_pose=True)
    return J, T[:, :3, 3], T[:, :3, :3]


def encode(p, R, task):
    """Pose encoding y for task '6d' (9 numbers) or '5d' (6 numbers)."""
    if task == '6d':
        return torch.cat([p, R[:, :, 0], R[:, :, 1]], dim=1)
    if task == '5d':
        return torch.cat([p, R[:, :, 2]], dim=1)
    raise ValueError(f'unknown task {task!r}')


def decode_rotation(y):
    """Rotation matrices (N, 3, 3) from 6-D encodings y (N, 9), by Gram-Schmidt
    on r1, r2. Exact for encodings produced by `encode`."""
    r1 = torch.nn.functional.normalize(y[:, 3:6], dim=1)
    r2 = y[:, 6:9] - (r1 * y[:, 6:9]).sum(1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(r2, dim=1)
    return torch.stack([r1, r2, torch.cross(r1, r2, dim=1)], dim=2)


def tangent_basis(a):
    """Two unit vectors (N, 3) each, orthogonal to the unit vectors a (N, 3) and
    to each other. Depends on a only, never on the configuration."""
    e = torch.zeros_like(a)
    idx = a.abs().argmin(dim=1)
    e[torch.arange(len(a)), idx] = 1.0
    u1 = torch.nn.functional.normalize(torch.cross(a, e, dim=1), dim=1)
    u2 = torch.cross(a, u1, dim=1)
    return u1, u2


def pa_jacobian(theta6, u1, u2):
    """Jacobian of the flange (p, a) in joints 1-6, (N, 5, 6).

    theta6: (N, 6) joints 1-6 (joint 7 does not move p or a).
    u1, u2: (N, 3) orthonormal tangent basis at the target axis a*. Rows are
    dp (3) and u1.da, u2.da with da = omega x a: an orthonormal chart of
    R^3 x S^2 at a*, so on the solution set (a = a*) det(J J^T) is the
    squared volume factor of the map, as the coarea formula needs.
    Also returns (p, a) at theta6."""
    q = torch.cat([theta6, torch.zeros_like(theta6[:, :1])], dim=1)
    J, p, R = jacobian(q)
    a = R[:, :, 2]
    Jv, Jw = J[:, :3, :6], J[:, 3:, :6]
    da = torch.cross(Jw, a[:, :, None].expand_as(Jw), dim=1)   # columns: omega_j x a
    rows = torch.cat([Jv, torch.einsum('ni,nij->nj', u1, da)[:, None], torch.einsum('ni,nij->nj', u2, da)[:, None]], dim=1)
    return rows, p, a


def roll_of(theta6, R_target):
    """Joint 7 angle that turns the flange from its q7 = 0 orientation at theta6
    into R_target, assuming the flange z axis already matches. Returns (N,) in
    (-pi, pi]."""
    q = torch.cat([theta6, torch.zeros_like(theta6[:, :1])], dim=1)
    _, R0 = fk(q)
    M = R0.transpose(1, 2) @ R_target   # = Rz(q7) on the solution set
    return torch.atan2(M[:, 1, 0], M[:, 0, 0])
