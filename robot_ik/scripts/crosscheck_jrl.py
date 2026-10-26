# SPDX-License-Identifier: MIT
"""Write a forward-kinematics fixture from Jrl's FR3, for tests/test_robot_ik_crosscheck.py.

Run once, in a separate environment that has Jrl (it needs klampt and changes
torch's global default device on import, so it is not a dependency here):

    uv venv --python 3.12 jrl_env
    VIRTUAL_ENV=jrl_env uv pip install "jrl @ git+https://github.com/jstmn/Jrl@5032d22db0608d36729c41d974d6bc8f9c5b41cf"
    jrl_env/bin/python robot_ik/scripts/crosscheck_jrl.py

Jrl's FR3 URDF comes from franka_ros (January 2025); its end effector is
`fr3_hand`, which sits at the flange `fr3_link8` rotated by -pi/4 about z. The
configurations are drawn inside the intersection of both URDFs' joint limits.
"""

from pathlib import Path

import numpy as np

JRL_COMMIT = '5032d22db0608d36729c41d974d6bc8f9c5b41cf'
OUT = Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / 'jrl_fr3_fk.npz'


def main():
    import torch
    from jrl.robots import Fr3

    robot = Fr3()
    rng = np.random.RandomState(0)
    lo = np.maximum(np.array([l for l, _ in robot.actuated_joints_limits]),
                    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
    hi = np.minimum(np.array([h for _, h in robot.actuated_joints_limits]),
                    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
    q = lo + (hi - lo) * rng.rand(64, 7)
    T = robot.forward_kinematics(torch.as_tensor(q, dtype=torch.float32), return_quaternion=False)
    T = T.detach().cpu().numpy().astype(np.float64)
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, q=q, T_hand=T, jrl_commit=JRL_COMMIT, end_effector=robot.end_effector_link_name)
    print(f'wrote {OUT}: {len(q)} configurations, end effector {robot.end_effector_link_name}')


if __name__ == '__main__':
    main()
