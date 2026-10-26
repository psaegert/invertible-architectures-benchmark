# SPDX-License-Identifier: MIT
"""Write the committed FR3 conditions: 1000 test poses (TEST_SEED, as for the toy
benchmarks) and 200 validation poses (VAL_SEED) per task, under metrics/conditions.

    python -m robot_ik.scripts.make_conditions

Both tasks use the same configurations x_true ~ prior; only the encoding of the
pose differs.
"""

from metrics.benchmarks import TEST_SEED, conditions_path, save_test_conditions

VAL_SEED = 20210127
N_VAL = 200


def main():
    for task in ('fr3_6d', 'fr3_5d'):
        for split, n, seed in (('test', 1000, TEST_SEED), ('val', N_VAL, VAL_SEED)):
            y, _ = save_test_conditions(task, split, n, seed)
            print(f'{task} {split}: {y.shape} -> {conditions_path(task, split)}')


if __name__ == '__main__':
    main()
