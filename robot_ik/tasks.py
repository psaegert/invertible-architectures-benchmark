# SPDX-License-Identifier: MIT
"""The two FR3 benchmark problems, with the interface of data/toy_data.py.

    fr3_6d   x = 7 joint angles (rad), y = flange pose [p, r1, r2] (9 numbers)
    fr3_5d   x = 7 joint angles (rad), y = flange position and z axis [p, a]
             (6 numbers); roll about a is free, so joint 7 is unconstrained

Prior: uniform within the URDF joint limits, no self-collision filtering.
Registered in data.toy_data.MODELS, so the existing training and evaluation
scripts accept `--problem fr3_6d` / `fr3_5d`. Torch and pytorch_kinematics are
imported on first use, so importing data.toy_data stays light.
"""

import numpy as np


class _Fr3Task:
    n_parameters = 7
    task = None

    def sample_prior(self, n, rng=np.random):
        from robot_ik.kinematics import sample_uniform
        return sample_uniform(n, rng)

    def forward_process(self, x, strict=True, chunk=200_000):
        """Pose encodings y (n, n_observations) for configurations x (n, 7), float64.
        `strict` is accepted for interface compatibility: f is defined everywhere."""
        import torch
        from robot_ik.kinematics import encode, fk
        x = np.asarray(x, dtype=np.float64)
        out = []
        for s in range(0, len(x), chunk):
            p, R = fk(torch.as_tensor(x[s:s + chunk]))
            out.append(encode(p, R, self.task).numpy())
        return np.concatenate(out) if out else np.empty((0, self.n_observations))

    def ground_truth(self, y_star, n_gt, seed, verbose=True):
        """Exact posterior samples, one (n_gt, 7) array per y* (robot_ik/ground_truth.py)."""
        from robot_ik.ground_truth import posterior_samples
        return posterior_samples(self.task, y_star, n_gt=n_gt, seed=seed, verbose=verbose)

    def task_metrics(self, x, y_star, gt):
        """Pose errors, joint-limit violations and coverage (robot_ik/metrics.py)."""
        from robot_ik.metrics import task_metrics
        return task_metrics(self.task, x, y_star, gt)


class Fr3Pose6D(_Fr3Task):
    """Full flange pose: a 1-D set of solutions per pose."""
    name = 'fr3-pose-6d'
    task = '6d'
    n_observations = 9


class Fr3Axis5D(_Fr3Task):
    """Flange position and z axis: a 1-D set in joints 1-6 times a free joint 7."""
    name = 'fr3-axis-5d'
    task = '5d'
    n_observations = 6
