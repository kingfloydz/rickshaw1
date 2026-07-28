# G1 Rickshaw Slopes

A Mjlab reinforcement learning task in which a Unitree G1 pulls a two-wheeled rickshaw over 19 fixed slopes. Training uses a privileged PPO teacher (S0), online teacher-to-student distillation (S1), and student PPO fine-tuning (S2).

## Requirements

| Component | Version |
| --- | --- |
| Python | `>=3.10,<3.14` |
| PyTorch | `>=2.7.0` |
| MuJoCo | `3.10.0` |
| Mjlab | `1.5.3` |
| MuJoCo-Warp | `3.10.0.3` |
| RSL-RL | `5.4.0` |

Linux with an NVIDIA GPU is recommended for training. Install the package in editable mode because the runtime resolves `assets/` and `config/` from the repository root:

```bash
cd /inspire/hdd/project/leverage-robot/ky26212/slopes

PYTHON=/root/miniconda3/bin/python
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e "source/g1_rickshaw_lab[test]"
```

The `test` extra installs `pytest` and `trimesh`. Install Ruff separately when needed:

```bash
"$PYTHON" -m pip install ruff
```

## Quick Start

```bash
cd /inspire/hdd/project/leverage-robot/ky26212/slopes
PYTHON=/root/miniconda3/bin/python

"$PYTHON" scripts/validate_mjlab_assets.py
"$PYTHON" scripts/validate_static_initialization.py
"$PYTHON" scripts/train_pipeline.py
```

The pipeline defaults to 8192 environments and 6000/6000/800 iterations for S0/S1/S2. It passes each completed stage checkpoint to the next stage automatically.

## Registered Tasks

| Stage | Task ID | Algorithm | Default iterations | Reward curriculum |
| --- | --- | --- | ---: | --- |
| S0 | `Mjlab-G1-Rickshaw-Slopes-Teacher` | Privileged PPO teacher | 6000 | S0 |
| S1 | `Mjlab-G1-Rickshaw-Slopes-Distillation` | Online distillation | 6000 | None |
| S2 | `Mjlab-G1-Rickshaw-Slopes-Student` | Student PPO fine-tuning | 800 | S2 |

## Architecture
### Physical Model

- G1 exposes 29 controlled joints. Dex1 finger joints are fixed and excluded from the action space.
- The rickshaw mass is `35.04 kg`; wheel radius is `0.3 m`; wheel track is `0.756462 m`.
- Two G1 grasp sites are connected to two rickshaw hitch sites with MuJoCo `connect` equalities.
- G1-rickshaw contacts, fixed-gripper contacts, and robot visual-mesh contacts are disabled. Robot self-collision and physical ground contacts remain active.
- G1 actuators use the Mjlab 1.5.3 Unitree defaults, including position actuators, effort limits, reflected armature, 10 Hz natural frequency, and damping ratio 2.0.
- MuJoCo runs at `0.005 s` with decimation 4, giving a 50 Hz policy rate. Solver, line-search, and CCD iterations are 10, 20, and 50.

### Initialization and Terrain

`scripts/validate_static_initialization.py` solves the complete closed-chain model on flat ground and writes the certified state to `config/static_rest_poses.json`. The certificate stores the model signature, full `qpos`, actuator torques, equality error, support-height error, hitch height, acceleration error, and maximum normalized actuator torque.

Training uses 19 slopes from `-0.08` to `0.10 rad` in `0.01 rad` increments. Each reset template is derived from the same flat certificate. G1 keeps its root orientation and adjusts only the ankle-pitch joints to the plane; the rickshaw rotates about the hitch axis so both wheels remain on the slope without breaking the hand constraints.

### Manager-Based Environment

The environment derives from Mjlab's `unitree_g1_flat_env_cfg()` and configures these managers:

| Manager | Responsibility |
| --- | --- |
| Scene | G1, rickshaw, closed-chain constraints, foot/self/wheel contact sensors |
| Event | Startup initialization, domain randomization, static reset, per-step state updates |
| Command | Rickshaw linear and yaw velocity in the terrain-aligned frame |
| Observation | Actor current/history, teacher dynamic/static privilege, critic privilege |
| Action | 29D joint-position targets around the slope-specific static reference |
| Reward | Mjlab G1 Flat rewards plus rickshaw-specific terms |
| Curriculum | Weight schedules for eight rickshaw penalties |
| Termination | 20-second timeout or G1 tilt above 70 degrees |

Training defaults to 8192 environments with `6 m` spacing and 20-second episodes. Play defaults to one environment with a near-unbounded episode length. Observation noise, domain randomization, and reward curricula are disabled in play mode.

### Commands and Actions

The `twist` command resamples every `3-8 s`:

| Component | Range |
| --- | --- |
| `lin_vel_x` | `[-1.5, 2.0] m/s` |
| `lin_vel_y` | `0` |
| `ang_vel_z` | `[-0.7, 0.7] rad/s` |

Standing commands occupy 10% of environments and forward-only commands occupy 20%. Command tracking uses the rickshaw wheel/terrain frame.

Actions map directly to joint targets:

```text
joint_target = q_ref + normalized_action * G1_ACTION_SCALE
```

`q_ref` is the slope-specific static joint reference. Per-joint action scale is `0.25 * effort_limit / stiffness`.

### Observation Contract

The deployable actor observation is fixed at 98D:

| Slice | Width | Feature | Scale |
| --- | ---: | --- | ---: |
| `0:3` | 3 | G1 base linear velocity | 1.0 |
| `3:6` | 3 | G1 base angular velocity | 0.25 |
| `6:9` | 3 | Projected gravity | 1.0 |
| `9:11` | 2 | Linear and yaw command | 1.0 |
| `11:40` | 29 | `joint_position - q_ref` | 1.0 |
| `40:69` | 29 | Joint velocity | 0.05 |
| `69:98` | 29 | Previous action | 1.0 |

History contains only frames preceding the current observation. Supported history lengths are 61 and 91.

The teacher additionally receives:

- 11D dynamic history: rickshaw linear/yaw velocity, pitch, left/right wheel normal force, and separate left/right hand forces in terrain tangent/lateral/normal axes.
- 10D episode-static privilege: torso mass, rickshaw mass and 3D COM, rolling resistance, terrain friction, left/right wheel damping, and terrain slope.

The critic receives the clean 98D current observation and 35D raw privilege. The privileged vector combines teacher static/current dynamic features, rickshaw acceleration, foot height, air time, contact state, and contact-force features.

### Policy Networks

Supported latent dimensions are:

```text
4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32
```

| Network | Architecture | Output |
| --- | --- | --- |
| Student context encoder | 98D history -> 48 channels -> four residual causal Conv1D blocks with dilation `1,2,4,8` -> linear | Latent context |
| Teacher encoder | `(98+11)D` history through the TCN; 10D static privilege through `10->16`; fused through `64->48->latent` | Latent context |
| Actor | `98D current + latent` -> ELU MLP `512,256,128` with learned Gaussian standard deviation | 29D action distribution |
| Critic | `98D current + 35D privilege` -> ELU MLP `512,256,128` | Scalar value |

The 61-frame model uses kernel size 5; the 91-frame model uses kernel size 7. Latent and history dimensions must match across training stages and checkpoint consumers.

### Domain Randomization

Nine physical parameters are sampled once per environment at startup:

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

`config/feasibility_envelope.yaml` is the source of these ranges and nominal calibration values.

## Rewards and Curricula

Curriculum entries below show their final weights.

| Reward term | Weight | Curriculum |
| --- | ---: | --- |
| `track_linear_velocity` | 2.0 | No |
| `track_angular_velocity` | 2.0 | No |
| `upright` | 0.1 | No |
| `pose` | 0.5 | No |
| `body_ang_vel` | -0.05 | No |
| `angular_momentum` | -0.02 | No |
| `dof_pos_limits` | -1.0 | No |
| `action_rate_l2` | -0.1 | No |
| `air_time` | 0.0 | No |
| `foot_clearance` | -2.0 | No |
| `foot_swing_height` | -0.25 | No; target `0.08 m` |
| `foot_slip` | -0.1 | No |
| `soft_landing` | -0.00001 | No |
| `self_collisions` | -1.0 | No |
| `arm_joint_velocity_l2` | -0.0015 | No |
| `rickshaw_forward_acceleration_l2` | -0.05 | Dynamics group |
| `rickshaw_pitch_angular_acceleration_l2` | -0.01 | Dynamics group |
| `rickshaw_yaw_angular_acceleration_l2` | -0.01 | Dynamics group |
| `rickshaw_pitch_angular_velocity_l2` | -1.0 | Dynamics group |
| `rickshaw_absolute_pitch_deviation_l2` | -0.5 | Dynamics group |
| `peak_force` | -3.0 | Dynamics group |
| `rickshaw_g1_relative_position_l2` | -4.0 | Relative-pose group |
| `rickshaw_g1_relative_yaw_l2` | -0.6 | Relative-pose group |
| `rickshaw_wheel_slip_l2` | -0.1 | No; always final weight |
| `hitch_height_recovery_l2` | -0.25 | No |

`peak_force` uses a cubic penalty with a 10 N soft limit and 50 N hard limit. Relative position applies 4x weight to axle-direction error. Hitch-height recovery uses a `0.05 m` deadband and `0.05 m` scale.

S0 curriculum:

| Iteration | Six dynamics terms | Two relative-pose terms |
| ---: | ---: | ---: |
| 0 | 0.04 | 0.08 |
| 300 | 0.12 | 0.20 |
| 600 | 0.30 | 0.40 |
| 900 | 0.50 | 0.60 |
| 1200 | 0.75 | 0.80 |
| 1500 | 1.00 | 1.00 |

S2 curriculum:

| Iteration | Six dynamics terms | Two relative-pose terms |
| ---: | ---: | ---: |
| 0 | 0.05 | 0.10 |
| 100 | 0.20 | 0.25 |
| 200 | 0.50 | 0.50 |
| 300 | 0.75 | 0.75 |
| 400 | 1.00 | 1.00 |

S1 and play configurations do not apply a reward-weight curriculum.

## Training Stages

### S0: Privileged Teacher PPO

The teacher actor uses observation history, dynamic privilege history, and static privilege. The critic uses clean current observations and raw privilege. PPO defaults are 24 rollout steps, 5 epochs, 4 mini-batches, learning rate `1e-3`, adaptive KL, `gamma=0.99`, `lambda=0.95`, and entropy coefficient `0.01`.

### S1: Online Distillation

`DistillationRunner` loads the S0 actor as the teacher, copies its policy trunk and policy observation normalizer into the student, and collects trajectories using student actions. Deterministic teacher actions are the MSE targets. Distillation uses learning rate `1e-3`, one epoch, and gradient length 15.

### S2: Student PPO

Fresh S2 initialization loads the complete S1 `student_state_dict` into the actor and the S0 `critic_state_dict` into the critic. A new PPO optimizer is created. Resume mode instead restores a complete S2 checkpoint and cannot be combined with `--teacher` or `--context`.

## Commands

All commands below run from the repository root and assume:

```bash
PYTHON=/root/miniconda3/bin/python
```

### Validate Assets

```bash
"$PYTHON" scripts/validate_mjlab_assets.py
```

Default report: `outputs/validation/mjlab_assets.json`.

```bash
"$PYTHON" scripts/validate_mjlab_assets.py \
  --output outputs/validation/assets_custom.json
```

### Solve and Certify the Static Pose

```bash
"$PYTHON" scripts/validate_static_initialization.py
```

The default command updates `config/static_rest_poses.json`, which is the file used by training. To write a separate report:

```bash
"$PYTHON" scripts/validate_static_initialization.py \
  --output outputs/validation/static_rest_pose.json
```

### Render the Certified Pose

```bash
"$PYTHON" scripts/render_reset_poses.py \
  --view side \
  --width 960 \
  --height 720

"$PYTHON" scripts/render_reset_poses.py \
  --view front \
  --output-dir outputs/reset_poses
```

Outputs are `reset_pose_flat.png` and `reset_pose_front.png` under `outputs/reset_poses/` by default.

### Run the Full Pipeline

```bash
"$PYTHON" scripts/train_pipeline.py \
  --latent-dim 4 \
  --history-length 61 \
  --rollout-steps 24 \
  --num-envs 8192 \
  --s0-iterations 6000 \
  --s1-iterations 6000 \
  --s2-iterations 800 \
  --seed 42 \
  --log-root logs/rsl_rl \
  --gpu-ids 0
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--latent-dim` | 16 | Shared latent width |
| `--history-length` | 61 | Shared history length; 61 or 91 |
| `--rollout-steps` | 24 | Steps per environment per update |
| `--num-envs` | 8192 | Parallel environments |
| `--s0-iterations` | 6000 | S0 iterations |
| `--s1-iterations` | 6000 | S1 iterations |
| `--s2-iterations` | 800 | S2 iterations |
| `--seed` | 42 | Environment and agent seed |
| `--log-root` | `logs/rsl_rl` | Experiment root |
| `--gpu-ids` | `0` | GPU list, `all`, or `None` |

Each stage runs in a separate subprocess. A failed stage stops the pipeline.

### Run Stages Separately

```bash
"$PYTHON" scripts/train_teacher.py \
  --latent-dim 4 \
  --num-envs 8192 \
  --max-iterations 6000 \
  --gpu-ids 0

TEACHER=/absolute/path/to/s0_teacher_checkpoint.pt

"$PYTHON" scripts/train_context.py \
  --teacher "$TEACHER" \
  --latent-dim 4 \
  --num-envs 8192 \
  --max-iterations 6000 \
  --gpu-ids 0

CONTEXT=/absolute/path/to/s1_context_checkpoint.pt

"$PYTHON" scripts/finetune_student.py \
  --teacher "$TEACHER" \
  --context "$CONTEXT" \
  --latent-dim 4 \
  --num-envs 8192 \
  --max-iterations 800 \
  --gpu-ids 0
```

The stage scripts share `--task`, `--experiment-dir`, `--latent-dim`, `--history-length`, `--rollout-steps`, `--num-envs`, `--max-iterations`, `--seed`, and `--gpu-ids`. The registered task default should normally be retained.

### Resume S0 or S2

```bash
"$PYTHON" scripts/train_teacher.py \
  --resume-checkpoint /absolute/path/to/s0_teacher_checkpoint.pt \
  --latent-dim 4 \
  --max-iterations 6000

"$PYTHON" scripts/finetune_student.py \
  --resume-checkpoint /absolute/path/to/s2_student_checkpoint.pt \
  --latent-dim 4 \
  --max-iterations 800
```

`--max-iterations` is the number of iterations executed by that `runner.learn()` call. S1 has no resume option.

### Select Compute Devices

```bash
# Multiple GPUs
"$PYTHON" scripts/train_pipeline.py --latent-dim 16 --gpu-ids 0 1

# All available GPUs
"$PYTHON" scripts/train_pipeline.py --latent-dim 16 --gpu-ids all

# CPU debug run
"$PYTHON" scripts/train_teacher.py \
  --num-envs 16 \
  --max-iterations 2 \
  --gpu-ids None
```

Mjlab uses `torchrunx` for multi-GPU launches.

### Monitor Training

```bash
"$PYTHON" -m tensorboard.main \
  --logdir logs/rsl_rl \
  --port 6006 \
  --bind_all
```

### Play a Student Checkpoint

```bash
STUDENT=/absolute/path/to/s2_student_checkpoint.pt

"$PYTHON" scripts/play.py \
  Mjlab-G1-Rickshaw-Slopes-Student \
  --checkpoint-file "$STUDENT" \
  --viewer viser \
  --num-envs 1
```

Use one fixed slope:

```bash
"$PYTHON" scripts/play.py \
  Mjlab-G1-Rickshaw-Slopes-Student \
  --slope 0.05 \
  --checkpoint-file "$STUDENT" \
  --viewer viser
```

`--slope` must be one of `-0.08,-0.07,...,0.09,0.10`. Without it, environments are distributed across the configured slopes. The one-environment default receives `-0.08 rad`; use `--num-envs 19` to inspect all slopes at once.

Record a video:

```bash
"$PYTHON" scripts/play.py \
  Mjlab-G1-Rickshaw-Slopes-Student \
  --checkpoint-file "$STUDENT" \
  --video True \
  --video-length 500 \
  --video-width 1280 \
  --video-height 720
```

Videos are written to `<checkpoint_run>/videos/play/`.

Run without a checkpoint:

```bash
"$PYTHON" scripts/play.py Mjlab-G1-Rickshaw-Slopes-Student --agent zero
"$PYTHON" scripts/play.py Mjlab-G1-Rickshaw-Slopes-Student --agent random
```

`play.py` currently builds the registered default actor (`latent_dim=16`, `history_length=61`). Trained playback therefore requires a checkpoint with those dimensions. Other dimensions can be trained and exported, but strict playback will reject their parameter shapes.

### Export JIT and ONNX Policies

```bash
# Default 16D/61-frame checkpoint
"$PYTHON" scripts/export_student.py \
  --checkpoint "$STUDENT"

# Non-default checkpoint
"$PYTHON" scripts/export_student.py \
  --checkpoint "$STUDENT" \
  --latent-dim 4 \
  --history-length 61 \
  --output-dir outputs/exported_student \
  --device cuda:0
```

Without `--output-dir`, files are written beside the checkpoint:

```text
exported/policy.pt
exported/policy.onnx
```

The exported model accepts `current [N,98]` and `history [N,H,98]`, and returns `actions [N,29]`.

### Test and Lint

```bash
PYTHONPATH=source/g1_rickshaw_lab \
  "$PYTHON" -m pytest -q

"$PYTHON" -m ruff check \
  scripts \
  source/g1_rickshaw_lab/g1_rickshaw_lab \
  tests
```

### CLI Help

```bash
"$PYTHON" scripts/validate_mjlab_assets.py --help
"$PYTHON" scripts/validate_static_initialization.py --help
"$PYTHON" scripts/render_reset_poses.py --help
"$PYTHON" scripts/train_pipeline.py --help
"$PYTHON" scripts/train_teacher.py --help
"$PYTHON" scripts/train_context.py --help
"$PYTHON" scripts/finetune_student.py --help
"$PYTHON" scripts/play.py Mjlab-G1-Rickshaw-Slopes-Student --help
"$PYTHON" scripts/export_student.py --help
```

## Outputs

```text
logs/rsl_rl/
  g1_rickshaw_teacher/<timestamp>_s0/
    model_<iteration>.pt
    params/env.yaml
    params/agent.yaml
  g1_rickshaw_context/<timestamp>_s1/
    model_<iteration>.pt
  g1_rickshaw_student/<timestamp>_s2/
    model_<iteration>.pt
    exported/policy.pt
    exported/policy.onnx

outputs/
  validation/mjlab_assets.json
  reset_poses/reset_pose_flat.png
  reset_poses/reset_pose_front.png
  nan_dumps/
```

RSL-RL saves every 50 iterations and once at the end of each training call. The final filename is not guaranteed to be `model_<max_iterations>.pt`; use the path printed by the pipeline or training log.

## Repository Layout

```text
assets/
  g1_dex1/                       G1 29-DoF and fixed Dex1 URDF/meshes
  rickshaw/                      Rickshaw URDF and meshes
config/
  feasibility_envelope.yaml      Domain-randomization ranges and calibration
  static_rest_poses.json         Certified flat static state
scripts/
  validate_mjlab_assets.py       Validate assets, assembly, collision, and hitch geometry
  validate_static_initialization.py
                                  Solve and certify static equilibrium
  render_reset_poses.py          Render the certified qpos directly in MuJoCo
  train_pipeline.py              Run S0 -> S1 -> S2
  train_teacher.py               Train or resume S0
  train_context.py               Run S1 online distillation
  finetune_student.py            Initialize or resume S2
  play.py                        Mjlab play entry with fixed-slope support
  export_student.py              Export JIT and ONNX policies
source/g1_rickshaw_lab/g1_rickshaw_lab/
  assets/                        MuJoCo/Mjlab asset builders
  configuration.py              Feasibility-envelope parser and validation
  static_equilibrium.py          Static certificate schema and I/O
  policy_schema.py               Observation, action, latent, and history ABI
  rl/                            Encoders, actor/critic, and RSL-RL adapters
  tasks/manager_based/rickshaw_velocity/
    closed_chain.py              G1/rickshaw assembly and hand equalities
    sloped_reset.py              Reset templates for 19 slopes
    terrain.py                   Slope assignment and terrain frames
    env_cfg.py                   Manager-based environment and reward curricula
    mjlab_actions.py             Static-reference joint-position action
    mjlab_commands.py            Rickshaw velocity command
    mjlab_events.py              Startup, reset, and step state updates
    mjlab_mdp.py                 Mjlab observation/reward adapters
    mdp/                         Dynamics, observation, and reward kernels
    agents/rsl_rl_cfg.py         PPO/distillation model and optimizer config
    agents/runners.py            Checkpoint handoff at the runner boundary
    registration.py              Mjlab task registration
tests/                            Geometry, reset, dynamics, policy, and checkpoint tests
```

## Compatibility Rules

- Use identical `latent_dim` and `history_length` values in S0, S1, S2, resume, and export commands.
- Fresh S2 requires both an S0 teacher checkpoint and an S1 context checkpoint.
- S2 resume cannot be combined with `--teacher` or `--context`.
- Re-run asset and static-pose validation after changing URDFs, mesh transforms, equalities, or actuators.
