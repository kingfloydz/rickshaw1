# G1 Rickshaw Mjlab Implementation Guide

The task is locked to MuJoCo 3.10.0, Mjlab 1.5.3, MuJoCo-Warp 3.10.0.3,
and RSL-RL 5.4.0. The registered tasks are:

- `Mjlab-G1-Rickshaw-Slopes-Teacher`
- `Mjlab-G1-Rickshaw-Slopes-Distillation`
- `Mjlab-G1-Rickshaw-Slopes-Student`

## Physical Specification

The rickshaw wheel diameter is 0.6 m and each wheel center remains 0.3 m along
the ground normal. The rickshaw center of mass is shifted rearward by
0.02 m.

The body-mesh tow points are `(0.276, -1.664929, 0.180746)` and
`(-0.276, -1.664929, 0.180746)` in the source STL frame. Fixed gripper sites
are connected to the corresponding rickshaw sites by two MuJoCo site-connect
equalities. The crossbar can rotate in the fixed claws, and all G1-rickshaw
collisions are disabled because contact between rigidly connected bodies would
conflict with the hand constraints. The fixed gripper collision geoms are also
disabled. Only the G1 URDF's group-0 collision geoms contact the ground; its
group-1 render meshes never participate in physics. The active geoms otherwise
use Mjlab's full-collision setup: self-collision is enabled, foot contacts use
three constraint dimensions and friction 0.6, and other contacts use one.

The six G1 actuator groups use Unitree's open-source Mjlab defaults: MuJoCo
built-in position actuators, 10 Hz natural frequency, damping ratio 2.0,
motor-specific reflected armature, and the published effort limits. Waist
roll/pitch and both ankle axes use the official doubled-5020 approximation.

## Initialization

At startup, MuJoCo inverse dynamics loads one certified flat-ground pose. The
12 leg and three waist joints use Unitree G1's default pose exactly; only the
14 arm joints and the two floating bases are optimized. The solve constrains
the hitch to 0.75-0.95 m, keeps a 5 mm optimization margin at each boundary,
and selects the valid pose with the smallest maximum normalized joint torque.
The current certified hitch height is 0.750492 m. Initial constraint error,
support error, acceleration, contact force, and actuator torque are certified.
Each sloped environment keeps the G1 root orientation from this same
flat-ground solution and changes only the two ankle-pitch joints to align the
foot soles. The rickshaw rotates about the hitch line so its wheels meet the
same slope without moving either hand connection; all other G1 joints remain
unchanged. The policy reference is `q_static` with `-terrain_slope` added to
both ankle-pitch joints. Actions map directly to
`q_ref + normalized_action * G1_ACTION_SCALE`; there is no torque-derived
position offset, action filter, or action clipping.

Run the physical validations before training:

```bash
python scripts/validate_mjlab_assets.py
python scripts/validate_static_initialization.py
```

## Training And Playback

```bash
python scripts/train_pipeline.py --latent-dim 8

# Or run each stage separately:
python scripts/train_teacher.py --latent-dim 8
python scripts/train_context.py --teacher <teacher.pt> --latent-dim 8
python scripts/finetune_student.py --teacher <teacher.pt> --context <distillation.pt> --latent-dim 8
python scripts/play.py Mjlab-G1-Rickshaw-Slopes-Student --checkpoint-file <student.pt>
python scripts/export_student.py --checkpoint <student.pt> --latent-dim 8
```

All stages use the checkpoint dictionaries produced by Mjlab and RSL-RL.
Distillation runs online through RSL-RL's `DistillationRunner`; student actions
drive the environment and deterministic teacher actions are the targets. Fresh
student PPO training loads the distilled `student_state_dict` and the teacher
`critic_state_dict` directly, with a new optimizer. Checkpoints are written to
timestamped run directories as `model_<iteration>.pt`. Non-default context and
history dimensions must be passed explicitly to every separately launched stage.
The pipeline command passes both dimensions and the generated checkpoints through
all three stages automatically.

S0 reward-weight curriculum stages are 0/300/600/900/1200 iterations; S2 stages
are 0/100/200/300/400. S1 has no reward curriculum. Wheel slip always uses its
final weight.

The Mjlab runtime owns 19 fixed slopes from -0.08 to 0.10 rad, startup-fixed nine-parameter domain
randomization, online dynamics diagnostics, observations, rewards, and RSL-RL
rollout state. The command observation contains only rickshaw `lin_vel_x` and
`ang_vel_z`; their tracking rewards use the rickshaw axle and terrain-normal
frame. The 98-D actor observation includes the command and measured rickshaw
velocities, G1 base linear/angular velocity, and the normalized previous action.
The clean critic observation adds official foot height, air time, contact state,
and signed-log contact force terms. Rewards otherwise use the Mjlab 1.5.3 G1
flat definitions, with upright weight 0.2, lower-body/waist-only pose tracking,
foot swing target 0.08 m, and the two rickshaw hitch-height terms.

The fixed rickshaw velocity command ranges, play episode/ranges, 24-step rollout, 50-iteration
checkpoint interval, S0/S1/S2 iteration defaults of 6,000/6,000/800, actor/critic MLPs, empirical
normalization, Gaussian standard deviation, and PPO hyperparameters match Mjlab
1.5.3 G1 Flat. There is no secondary simulator runtime path or runtime reward
override.
