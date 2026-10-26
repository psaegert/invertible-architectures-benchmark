# Provenance of fr3.urdf

- Source: [franka_description](https://github.com/frankarobotics/franka_description) by Franka Robotics GmbH, Apache License 2.0 (`LICENSE` and `NOTICE` in this folder).
- Commit: `7aeeddc449edf8d62b594f9e36a81da53e7796f9` (release 2.9.0, 2026-08-17).
- Input: `robots/fr3/fr3.urdf.xacro` with `hand:=false`, expanded with the pip `xacro` package.
- Modification: reduced to the kinematic chain `base` → `fr3_link8` (the flange): link names, joint origins, axes and limits only. Visuals, collisions, inertials, accelerometer frames and self-collision sub-links were removed.
- Regenerate with `python -m robot_ik.scripts.make_urdf path/to/franka_description` (the script checks the commit).

Values checked after generation: the flange sits at (0.088, 0, 0.926) m at the zero configuration, and the joint limits equal `robots/fr3/joint_limits.yaml`.
