# FR3 inverse kinematics benchmark (`robot_ik/`)

Two benchmark problems on the 7-joint Franka FR3, plugged into the shared harness like the toy benchmarks: train with `--problem fr3_6d` or `fr3_5d`, score with `metrics.evaluate`. The ground truth is the exact posterior, so Err_post (MMD) and calibration mean the same as on the toy problems. Everything here is open source; the robot model is Franka's official description.

## The two tasks

| Problem | x (7) | y | Solutions per pose |
|---|---|---|---|
| `fr3_6d` | joint angles (rad) | flange pose: position (m) and the first two columns of the rotation matrix, 9 numbers | a curve (1-D) |
| `fr3_5d` | joint angles (rad) | flange position (m) and z axis (unit vector), 6 numbers; roll about the axis is free | a curve in joints 1–6 times a free joint 7 |

- **End effector:** the flange `fr3_link8` (no hand), 0.107 m past joint 7 along its axis. Joint 7 spins the flange about that axis, so the flange position and z axis depend on joints 1–6 only. In the 5-D task joint 7 is therefore independent of y and uniform within its limits; a model has to learn that.
- **Prior:** uniform within the URDF joint limits (`robot_ik.kinematics.joint_limits()`). No self-collision filtering.
- **Conditions:** 1000 test poses (`metrics/conditions/fr3_*.npz`, the toy TEST_SEED) and 200 validation poses (`fr3_*_val.npz`). Select models on `split='val'`, report on `split='test'`.
- **Angles:** models must return angles as values inside the joint limits. Every joint range is shorter than 2π, so that value is unique; an angle outside the limits counts as a violation, even if it is the same direction modulo 2π.

## Plugging in a model

```python
from metrics.evaluate import evaluate
evaluate(sample, 'fr3_6d', name='fmpe_s0', split='val')   # model selection
evaluate(sample, 'fr3_6d', name='fmpe_s0')                # the reported number
```

`sample(y_star, n)` returns `(M, n, 7)` joint angles in radians, exactly as for the toy problems with `dim_x = 7`. Training data: `data.toy_data.make_dataset('fr3_6d', n, seed)` (float32 `x (n, 7)`, `y (n, 9)`).

Places in the existing model code that assume the toy problems and need adapting: `GT_SIGMA[problem]` lookups (FFF, FIF), `x_dims=4` in `train_fif.py`, `ANGLE_DIMS` keyed by benchmark name in `models/fff.py`, and the `kin`/`bal` run-name prefixes in the plot scripts. The INN with `z = x_dims - y_dims` cannot be built for the 9-number pose (z would be negative); the cINN can.

## What the evaluator reports

On top of the toy numbers (Err_post, Err_resim on the encodings, timing):

| Key | Meaning |
|---|---|
| `pos_err_mm` | flange position error per sample: median, mean, p95 |
| `rot_err_deg` (6-D) / `axis_err_deg` (5-D) | geodesic rotation error, or angle between flange axes |
| `success_rate` | share of samples within 1 mm and 1°, and within 10 mm and 5°, and inside the joint limits |
| `joint_limit_violation_fraction` | samples with any joint outside its limits (nothing is clamped) |
| `coverage` | share of ground-truth draws with a model sample within δ (rad, Euclidean in joint space) |

The npz files also hold the per-pose medians, violation shares and coverage. `metrics.calibration` works unchanged.

**Oracle row.** `robot_ik.ground_truth.oracle_sampler` returns a second, independent draw of the ground truth. Scored as a model, it gives the noise floor of Err_post and coverage: a model cannot do better than the oracle except by chance. The committed `metrics/results/fr3_*_oracle.json` hold it for the reference generation; after regenerating the sets, rerun `python -m robot_ik.scripts.validate_gt report fr3_6d` (and `fr3_5d`) to refresh it.

Oracle row on the 1000 test poses (the floor any model is compared against):

| | `fr3_6d` | `fr3_5d` |
|---|---|---|
| Err_post mean [q1, q3] | −0.0001 [−0.0008, 0.0005] | 0.0000 [−0.0003, 0.0002] |
| coverage (δ) | 0.995 (0.1 rad) | 0.982 (0.5 rad) |
| position error p95 | 7e-5 mm (float32 storage) | 7e-5 mm |
| success@1 mm, 1° | 1.000 | 1.000 |

Differences in Err_post below about 5e-4 are within this noise. The coverage radius is larger for the 5-D task because its solutions fill a surface (the curve times the free joint 7), so 1000 samples lie further apart.

## Ground truth

The flange position p and z axis a depend on joints 1–6 only, so all solutions of a pose lie on the curve C = {θ₁..₆ : p = p*, a = a*}: five constraints in six joints. `robot_ik/manifold.py` finds and samples C exactly:

1. **Discovery:** 10,000 seeds per pose, uniform on the whole torus, Levenberg–Marquardt on [p − p*, a − a*]. Seeds never come from the configuration that generated the pose.
2. **Tracing:** every component of C is a closed loop on the torus. From one solution per component, the loop is followed along the cofactor field c of the 5 × 6 Jacobian (RK4 in arclength, Newton corrector, step ≤ 0.02 rad).
3. **Weighting:** by the coarea formula the posterior on C has density ∝ prior / √det(JJᵀ) per unit length, and |c| = √det(JJᵀ) (Cauchy–Binet). Along dθ/dτ = c the posterior is therefore uniform in τ, and τ is integrated with Simpson's rule.
4. **Sampling:** τ uniform over all loops, Hermite interpolation, a Newton correction, then rejection of points outside the joint limits. 5-D task: joint 7 uniform within its limits. 6-D task: joint 7 is fixed by the remaining roll, and must be inside its limits too.

The sets for the committed conditions live in `metrics/cache/` (`gt_fr3_*.npz`: 56 MB per test set, 11 MB per validation set, each with the oracle draw and per-pose diagnostics) and are not in git. The first evaluation of a task generates its set; on a GPU that takes about 15 minutes per 1000 poses (6.5 minutes for the 200 validation poses), on a CPU hours. Floating-point differences between machines make regenerated draws differ slightly, so **generate each set once, share those files within the team, and score every model against the same files**. Either generate them up front:

```bash
python -m robot_ik.scripts.generate_gt fr3_6d --split test --device cuda
nohup robot_ik/scripts/run_gt_queue.sh GPU > gt_queue.log 2>&1 < /dev/null &   # all four sets
```

### Validation

Each check below is in `robot_ik/scripts/validate_gt.py` or the tests; the outputs are in `docs/robot_ik_validation/`.

| Check | Result |
|---|---|
| Tracer on an ellipse (analytic posterior) | mass within 2e-7 of π a b; phase uniform (KS) |
| Tracer on a planar 3-joint arm vs rejection sampling, box and Gaussian prior | not distinguishable (MMD permutation p > 0.01); sampling uniform in arclength instead, which drops the coarea weight, is rejected (p < 0.005) |
| Two loops of different mass | draws split by mass (this caught a bug in the first FR3 run, since fixed and regenerated) |
| FK against Jrl's independent FR3 model | flange pose within 2e-6 (Jrl's float32) |
| All loops closed, 1000 test poses per task | yes; worst closure error 1.4e-9 rad; 4–8 loops per pose (median 8) |
| Generating configuration on a traced loop (never used as a seed) | 1000 of 1000 per task: missed posterior mass below 0.3 % (95 %) |
| Twice the seeds and half the step, 20 poses | same loops; mass changes at most 1.3e-6 (median 1e-9) |
| Direct trace in all 7 joints (6 constraints, geometric Jacobian) vs the joint 1–6 curve, 10 poses | identical loop masses; samples not distinguishable (permutation p from 0.055 to 0.965) |
| Calibration of the ground truth against the generating configurations (1000 poses) | calibrated in every joint and jointly (signed error +0.005 for 6-D, +0.008 for 5-D) |

About 30 % of each pose's solution curve lies inside the joint limits (median; down to 0.6 % for some poses), which is why the posterior is not simply "the self-motion loop".

**IKFlow reference on the Panda.** IKFlow's public pretrained model `panda__full__lp191_5.25m`, evaluated with its own script (500 poses × 50 samples, latent scale 0.75, clamped to the joint limits), gives 6.85 mm and 2.34° mean error, matching the 6.83 mm and 2.28° in its repository history (the paper reports 7.72 mm and 2.81°). A reimplementation should reach that on the Panda before it moves to the FR3.

## Files

| Path | What |
|---|---|
| `robot_ik/kinematics.py` | FK, Jacobians, encodings (pytorch_kinematics on the URDF) |
| `robot_ik/tasks.py` | the two problems, registered in `data.toy_data.MODELS` |
| `robot_ik/manifold.py` | the generic fiber tracer and sampler |
| `robot_ik/ground_truth.py` | the FR3 fiber, generation, cache, oracle sampler |
| `robot_ik/metrics.py` | pose errors, joint limits, coverage |
| `robot_ik/assets/fr3/` | the URDF with Franka's LICENSE, NOTICE and PROVENANCE.md |
| `robot_ik/scripts/` | URDF generation, conditions, ground truth (+ queue), validation, the Jrl cross-check, the planar fixture |
| `tests/test_robot_ik_*.py` | kinematics, the Jrl cross-check, tasks in the harness, the tracer on problems with known answers |

## Licensing and provenance

`robot_ik/` is MIT-licensed (`robot_ik/LICENSE`). The URDF is generated from [franka_description](https://github.com/frankarobotics/franka_description) (Apache-2.0, commit `7aeeddc`, release 2.9.0) and reduced to the kinematic chain; see `robot_ik/assets/fr3/PROVENANCE.md`. Kinematics come from [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) 0.10.0 (MIT). [Jrl](https://github.com/jstmn/Jrl) (MIT) was used once, outside this repo, to cross-check the forward kinematics (`tests/fixtures/jrl_fr3_fk.npz`). IKFlow's code has no license (its LICENSE.txt reads `#TODO`): use it, but do not copy it into this repo.
