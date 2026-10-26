# SPDX-License-Identifier: MIT
"""Rejection-sampling reference for the planar 3R tracer test
(tests/test_robot_ik_manifold.py): two independent runs of 2000 posterior
draws for each prior, written to tests/fixtures/planar3r_rejection.npz.

    python -m robot_ik.scripts.make_planar_fixture

Takes a few minutes on a CPU; the test itself then only runs the tracer.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))

from test_robot_ik_manifold import PLANAR_CASES, rejection  # noqa: E402

OUT = ROOT / 'tests' / 'fixtures' / 'planar3r_rejection.npz'


def main():
    rng = np.random.RandomState(0)
    arrays = {}
    for name, (target, lower, upper, prior_sigma) in PLANAR_CASES.items():
        for run in (1, 2):
            arrays[f'{name}_{run}'] = rejection(np.asarray(target), lower, upper, prior_sigma, 2000, rng)
            print(f'{name} run {run} done', flush=True)
    np.savez(OUT, **arrays)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
