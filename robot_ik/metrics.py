# SPDX-License-Identifier: MIT
"""FR3-specific metrics, merged into metrics.evaluate's results.

Per model sample: position error (mm) and orientation error (deg): the
geodesic rotation angle for the 6-D task, the angle between flange axes for
the 5-D task. Joint-limit violations are counted on the raw samples (nothing
is clamped). Coverage: the fraction of ground-truth draws that have a model
sample within COVERAGE_DELTA[task] (rad, Euclidean in joint space). The radii
are set so that a second, independent ground-truth draw (the oracle) covers
about 98 % or more: 0.1 rad for the 6-D task (a curve), 0.5 rad for the 5-D
task (a curve times the free joint 7).
"""

import numpy as np
import torch

from robot_ik import kinematics as K

COVERAGE_DELTA = {'6d': 0.1, '5d': 0.5}  # rad; oracle coverage 0.994 / 0.982 (docs/robot_ik.md)
SUCCESS = {'1mm_1deg': (1.0, 1.0), '10mm_5deg': (10.0, 5.0)}


def pose_errors(task, x, y_star, chunk=200_000):
    """Position error (mm) and orientation error (deg), each (M, n), for
    samples x (M, n, 7) against conditions y_star (M, dim_y)."""
    M, n, _ = x.shape
    y = torch.as_tensor(np.repeat(np.asarray(y_star, dtype=np.float64), n, axis=0))
    q = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(M * n, 7))
    pos, ang = [], []
    for s in range(0, M * n, chunk):
        p, R = K.fk(q[s:s + chunk])
        t = y[s:s + chunk]
        pos.append(1000 * torch.linalg.norm(p - t[:, :3], dim=1))
        if task == '6d':
            d = torch.linalg.norm(R - K.decode_rotation(t), dim=(1, 2))      # = 2 sqrt(2) sin(angle / 2)
            ang.append(2 * torch.asin(torch.clamp(d / (2 * np.sqrt(2)), max=1.0)))
        else:
            d = torch.linalg.norm(R[:, :, 2] - t[:, 3:6], dim=1)            # = 2 sin(angle / 2)
            ang.append(2 * torch.asin(torch.clamp(d / 2, max=1.0)))
    return (torch.cat(pos).numpy().reshape(M, n), np.degrees(torch.cat(ang).numpy()).reshape(M, n))


def joint_limit_violations(x):
    """(M, n) bool: any joint outside its URDF limit."""
    lo, hi = K.joint_limits()
    return ((x < lo) | (x > hi)).any(-1)


def coverage(x, gt, delta):
    """Per y*: fraction of ground-truth draws with a model sample within delta."""
    out = np.empty(len(gt))
    for i, (a, b) in enumerate(zip(x, gt)):
        d = torch.cdist(torch.as_tensor(b, dtype=torch.float64), torch.as_tensor(a, dtype=torch.float64))
        out[i] = (d.min(dim=1).values <= delta).double().mean().item()
    return out


def _quantiles(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return {'median': float(np.median(v)), 'mean': float(v.mean()), 'p95': float(np.quantile(v, 0.95))}


def task_metrics(task, x, y_star, gt):
    """Returns (summary dict for the JSON, per-y* arrays for the npz)."""
    pos, ang = pose_errors(task, x, y_star)
    viol = joint_limit_violations(x)
    cov = coverage(x, gt, COVERAGE_DELTA[task])
    ang_key = 'rot_err_deg' if task == '6d' else 'axis_err_deg'
    summary = {
        'pos_err_mm': _quantiles(pos),
        ang_key: _quantiles(ang),
        'success_rate': {k: float(((pos <= mm) & (ang <= deg) & ~viol).mean()) for k, (mm, deg) in SUCCESS.items()},
        'joint_limit_violation_fraction': float(viol.mean()),
        'coverage': {'delta_rad': COVERAGE_DELTA[task], **_quantiles(cov)},
    }
    arrays = {'pos_err_mm_median': np.median(pos, axis=1), f'{ang_key}_median': np.median(ang, axis=1),
              'joint_limit_violation': viol.mean(axis=1), 'coverage': cov}
    return summary, arrays
