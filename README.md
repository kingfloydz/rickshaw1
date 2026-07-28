# G1 Rickshaw Slopes

A MjLab reinforcement-learning task in which a Unitree G1 pulls a two-wheeled
rickshaw over 19 fixed slopes. Training follows one main pipeline:

1. S0 trains a privileged PPO teacher.
2. S1 distills the teacher into a history-only student.
3. S2 fine-tunes the student with PPO.

## Requirements

| Component | Version |
| --- | --- |
| Python | `>=3.10,<3.14` |
| PyTorch | `>=2.7.0` |
| MuJoCo | `3.10.0` |
| MjLab | `1.5.3` |
| MuJoCo-Warp | `3.10.0.3` |
| RSL-RL | `5.4.0` |

Linux with an NVIDIA GPU is recommended for training. Install the package in
editable mode because assets and configuration are resolved from the repository:

```bash
cd /path/to/rickshaw1

PYTHON=/path/to/python
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e "source/g1_rickshaw_lab[test]"
```

The `test` extra installs Pytest. Install Ruff separately for linting:

```bash
"$PYTHON" -m pip install ruff
```

## Quick Start

```bash
cd /path/to/rickshaw1
PYTHON=/path/to/python

"$PYTHON" scripts/validate_static_initialization.py
"$PYTHON" scripts/train_pipeline.py --gpu-ids 0
```

The pipeline uses 8192 environments by default and runs 6000, 6000, and 800
iterations for S0, S1, and S2. Each completed checkpoint is passed to the next
stage automatically.

## Registered Tasks

| Stage | Task ID | Algorithm | Iterations | Curriculum |
| --- | --- | --- | ---: | --- |
| S0 | `Mjlab-G1-Rickshaw-Slopes-Teacher` | Privileged PPO | 6000 | S0 |
| S1 | `Mjlab-G1-Rickshaw-Slopes-Distillation` | Online distillation | 6000 | None |
| S2 | `Mjlab-G1-Rickshaw-Slopes-Student` | Student PPO | 800 | S2 |

## Architecture

### Physical Model

- G1 exposes 29 controlled joints; Dex1 finger joints are fixed.
- Two grasp sites are connected to the rickshaw hitches with MuJoCo `connect`
  equalities.
- G1 uses the MjLab 1.5.3 Unitree actuator configuration.
- The simulation timestep is `0.005 s` with decimation 4, giving a 50 Hz policy
  rate.
- Robot-rickshaw collisions and visual-mesh collisions are disabled while robot
  self-collision and physical ground contacts remain active.

`scripts/validate_static_initialization.py` solves the complete closed-chain
model on flat ground and writes the certified state to
`config/static_rest_poses.json`. The same certificate is used to build reset
templates for all slopes.

Training covers slopes from `-0.08` to `0.10 rad` in `0.01 rad` increments.
G1 remains upright and adjusts ankle pitch to the plane; the rickshaw is aligned
to keep both wheels on the slope without breaking the hand constraints.

### Manager-Based Environment

The environment derives from MjLab's `unitree_g1_flat_env_cfg()`:

| Manager | Project responsibility |
| --- | --- |
| Scene | G1, rickshaw, closed-chain equalities, and contact sensors |
| Event | Domain initialization, certified reset, and rolling resistance |
| Command | Rickshaw forward and yaw velocity in the terrain frame |
| Observation | Actor sequence, teacher privilege, and critic privilege |
| Action | Joint-position targets around the slope-specific static pose |
| Reward | MjLab G1 rewards plus rickshaw-specific terms |
| Curriculum | S0 and S2 schedules for rickshaw penalties |
| Termination | Timeout or excessive G1 tilt |

Training uses 8192 environments with `6 m` spacing. Play mode uses one
environment and disables observation noise, domain randomization, and reward
curricula.

### Commands and Actions

The `twist` command is resampled every `3-8 s`:

| Component | Range |
| --- | --- |
| Forward velocity | `[-1.5, 2.0] m/s` |
| Lateral velocity | `0` |
| Yaw velocity | `[-0.7, 0.7] rad/s` |

Actions are 29 normalized joint-position offsets:

```text
joint_target = q_ref + normalized_action * G1_ACTION_SCALE
```

`q_ref` is the certified slope-specific joint pose. Per-joint scale is
`0.25 * effort_limit / stiffness`.

### Observations and Models

The deployable actor observation is fixed at 98 dimensions:

| Slice | Width | Feature | Scale |
| --- | ---: | --- | ---: |
| `0:3` | 3 | G1 base linear velocity | 1.0 |
| `3:6` | 3 | G1 base angular velocity | 0.25 |
| `6:9` | 3 | Projected gravity | 1.0 |
| `9:11` | 2 | Forward and yaw command | 1.0 |
| `11:40` | 29 | Joint position error | 1.0 |
| `40:69` | 29 | Joint velocity | 0.05 |
| `69:98` | 29 | Previous action | 1.0 |

MjLab's observation manager stores the preceding 61 frames. The current frame
is kept separate from the history passed to the context encoder.

The teacher also receives dynamic rickshaw/contact history and episode-static
physical parameters. The critic receives the clean current actor observation
plus raw privileged state. The student uses only actor history.

The default model uses a 16D context:

| Network | Main structure |
| --- | --- |
| Student context | Four residual causal Conv1D blocks with dilations `1,2,4,8` |
| Teacher context | Actor and dynamic histories fused with static privilege |
| Actor | RSL-RL ELU MLP `512,256,128` and Gaussian distribution |
| Critic | RSL-RL ELU MLP `512,256,128` |

### Domain Randomization

Nine physical parameters are sampled once per environment:

| Parameter | Range |
| --- | --- |
| `torso.mass_delta` | `[-1.0, 3.0] kg` |
| `payload.mass` | `[-3.0, 3.0] kg` |
| `payload.com.x` | `[0.3, 0.9] m` |
| `payload.com.y` | `[-0.15, 0.15] m` |
| `payload.com.z` | `[0.45, 0.95] m` |
| `rolling_resistance.c_rr` | `[0.01, 0.03]` |
| `terrain.friction` | `[0.6, 1.1]` |
| `wheel.left_damping` | `[0.015, 0.025]` |
| `wheel.right_damping` | `[0.015, 0.025]` |

`config/feasibility_envelope.yaml` is the source of these ranges and nominal
calibration values.

## Rewards and Curricula

The task retains the MjLab G1 Flat reward set and changes only the weights or
parameters needed for towing. Project-specific terms are:

| Reward term | Final weight |
| --- | ---: |
| `track_linear_velocity` | 2.0 |
| `track_angular_velocity` | 2.0 |
| `arm_joint_velocity_l2` | -0.0015 |
| `rickshaw_forward_acceleration_l2` | -0.05 |
| `rickshaw_pitch_angular_acceleration_l2` | -0.01 |
| `rickshaw_yaw_angular_acceleration_l2` | -0.01 |
| `rickshaw_pitch_angular_velocity_l2` | -1.0 |
| `rickshaw_wheel_slip_l2` | -0.1 |
| `rickshaw_absolute_pitch_deviation_l2` | -0.5 |
| `rickshaw_g1_relative_position_l2` | -4.0 |
| `rickshaw_g1_relative_yaw_l2` | -0.6 |
| `peak_force` | -3.0 |
| `hitch_height_recovery_l2` | -0.25 |

Six dynamics penalties and two relative-pose penalties are ramped during S0 and
S2. The table values are multipliers of their final weights.

S0 curriculum:

| Iteration | Dynamics | Relative pose |
| ---: | ---: | ---: |
| 0 | 0.04 | 0.08 |
| 300 | 0.12 | 0.20 |
| 600 | 0.30 | 0.40 |
| 900 | 0.50 | 0.60 |
| 1200 | 0.75 | 0.80 |
| 1500 | 1.00 | 1.00 |

S2 curriculum:

| Iteration | Dynamics | Relative pose |
| ---: | ---: | ---: |
| 0 | 0.05 | 0.10 |
| 100 | 0.20 | 0.25 |
| 200 | 0.50 | 0.50 |
| 300 | 0.75 | 0.75 |
| 400 | 1.00 | 1.00 |

S1 and play mode do not apply reward-weight curricula.

## Workflows

All commands run from the repository root and assume:

```bash
PYTHON=/path/to/python
```

### Static Initialization

```bash
"$PYTHON" scripts/validate_static_initialization.py
```

This command updates the certified flat pose used by training. Re-run it after
changing geometry, actuators, or equality constraints.

### Train

Run the complete pipeline:

```bash
"$PYTHON" scripts/train_pipeline.py --gpu-ids 0
```

Common operational overrides are `--num-envs`, `--s0-iterations`,
`--s1-iterations`, `--s2-iterations`, `--seed`, `--log-root`, and
`--gpu-ids`. Use `--gpu-ids all` for all available GPUs or `None` for a
small CPU debug run.

Run stages separately:

```bash
"$PYTHON" scripts/train_teacher.py --gpu-ids 0

TEACHER=/absolute/path/to/s0_teacher_checkpoint.pt
"$PYTHON" scripts/train_context.py --teacher "$TEACHER" --gpu-ids 0

CONTEXT=/absolute/path/to/s1_context_checkpoint.pt
"$PYTHON" scripts/finetune_student.py \
  --teacher "$TEACHER" \
  --context "$CONTEXT" \
  --gpu-ids 0
```

Resume S0 or S2 with `--resume-checkpoint`. A fresh S2 run requires both the S0
teacher and S1 context checkpoints; a resumed S2 run uses only its S2 checkpoint.

### Play and Export

```bash
STUDENT=/absolute/path/to/s2_student_checkpoint.pt

"$PYTHON" scripts/play.py \
  Mjlab-G1-Rickshaw-Slopes-Student \
  --checkpoint-file "$STUDENT" \
  --viewer viser
```

Use `--slope 0.05` to select one configured slope. Without it, environments
are assigned across all 19 slopes.

Export the deployable student:

```bash
"$PYTHON" scripts/export_student.py --checkpoint "$STUDENT"
```

The exporter writes `exported/policy.pt` and `exported/policy.onnx` beside the
checkpoint. Both accept `current [N,98]` and `history [N,61,98]`, and return
`actions [N,29]`.

### Test and Lint

```bash
PYTHONPATH=source/g1_rickshaw_lab \
  "$PYTHON" -m pytest -q

"$PYTHON" -m ruff check \
  scripts \
  source/g1_rickshaw_lab/g1_rickshaw_lab \
  tests
```

## Outputs

```text
logs/rsl_rl/
  g1_rickshaw_teacher/<timestamp>_s0/model_<iteration>.pt
  g1_rickshaw_context/<timestamp>_s1/model_<iteration>.pt
  g1_rickshaw_student/<timestamp>_s2/model_<iteration>.pt

outputs/
  validation/
  reset_poses/
  nan_dumps/
```

RSL-RL saves periodically and at the end of each training call. Use the
checkpoint path printed by the pipeline rather than assuming the final filename.

## Repository Layout

```text
assets/
  g1_dex1/                         G1 URDF and fixed-Dex1 mesh assets
    meshes/                        G1 visual and collision meshes
  rickshaw/                        Rickshaw URDF and mesh assets

config/
  feasibility_envelope.yaml        Randomization ranges and calibration
  static_rest_poses.json           Certified flat-ground qpos and diagnostics

scripts/
  _project.py                      Source-layout import bootstrap
  _stage_training.py               Shared MjLab stage configuration
  _static_equilibrium_solver.py    Closed-chain static optimization
  validate_static_initialization.py
                                    Solve and save the static certificate
  train_pipeline.py                Orchestrate S0 -> S1 -> S2
  train_teacher.py                 Train or resume the S0 teacher
  train_context.py                 Run S1 online distillation
  finetune_student.py              Initialize or resume S2 PPO
  play.py                          Playback, slope selection, and recording
  export_student.py                Export TorchScript and ONNX students

source/g1_rickshaw_lab/
  pyproject.toml                   Build, Ruff, and Pytest configuration
  setup.py                         Package metadata and pinned dependencies
  g1_rickshaw_lab/
    __init__.py                    Package marker and public package metadata
    project_paths.py               Repository asset/config path resolution
    configuration.py               Feasibility schema parser and validation
    g1_motor_defaults.py           Ordered G1 actuator and action constants
    policy_schema.py               Policy observation/action ABI
    rickshaw_spec.py               Pure rickshaw mechanical specification
    static_equilibrium.py          Static-certificate schema, signature, and I/O

    assets/
      __init__.py                  Public asset configuration exports
      mujoco_spec.py               Shared MjSpec and collision helpers
      g1_dex1.py                   G1 URDF validation and MjLab robot config
      rickshaw.py                  Rickshaw URDF validation and entity config

    rl/
      __init__.py                  Public encoder and model exports
      context_encoder.py           Student causal temporal encoder
      teacher_model.py             Privileged teacher context encoder
      rsl_rl_models.py             RSL-RL actor/critic adapters and export wrappers

    tasks/manager_based/rickshaw_velocity/
      __init__.py                  Task IDs and public registration surface
      registration.py              MjLab environment/runner registrations
      env_cfg.py                   Scene, managers, rewards, and curricula
      closed_chain.py              G1-rickshaw assembly and hand equalities
      terrain.py                   Slope assignment, frames, and plane poses
      sloped_reset.py              Certified reset templates for all slopes
      mjlab_actions.py             Static-reference joint-position action
      mjlab_commands.py            Terrain-frame rickshaw velocity command
      mjlab_events.py              MjLab startup, reset, and step integration
      mjlab_mdp.py                 MjLab observation and reward term adapters

      mdp/
        __init__.py                Public task kernel exports
        dynamics.py                Kinematics, rolling resistance, and forces
        events.py                  Randomization and runtime-state kernels
        observations.py            Actor and privileged observation assembly
        rewards.py                 Rickshaw-specific reward kernels

      agents/
        __init__.py                Agent configuration and runner exports
        rsl_rl_cfg.py              S0/S1/S2 RSL-RL configurations
        runners.py                 Distillation and checkpoint handoff logic

tests/
  conftest.py                      Source-layout Pytest bootstrap
  test_checkpoint_flow.py          S1-to-S2 checkpoint handoff
  test_command_action_dynamics.py  Action ABI and dynamics kernels
  test_feasibility_configuration.py
                                    Feasibility schema validation
  test_mjlab_migration.py          Assets, assembly, curricula, and certificates
  test_observation_and_tcn.py      Observation order and temporal encoders
  test_online_distillation.py      RSL-RL integration and legacy checkpoints
  test_play_cli.py                 Playback-specific argument handling
  test_reward_kernels.py           Reward kernels and task reward contract
  test_sloped_reset.py             Slope reset geometry and constraints
  test_sloped_terrain.py           Terrain frames and wheel contacts
  test_training_pipeline.py        Three-stage command/checkpoint orchestration
```

## Compatibility Rules

- Keep model dimensions identical across S0, S1, S2, playback, and export.
- Fresh S2 initialization requires both an S0 teacher checkpoint and an S1
  context checkpoint.
- S2 resume cannot be combined with `--teacher` or `--context`.
- Re-run static initialization after changing URDFs, mesh transforms,
  equalities, collision masks, or actuators.
