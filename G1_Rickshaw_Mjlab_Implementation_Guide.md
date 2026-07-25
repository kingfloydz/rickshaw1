# G1 Rickshaw Mjlab Implementation Guide

The task is locked to MuJoCo 3.10.0, Mjlab 1.5.3, MuJoCo-Warp 3.10.0.3,
and RSL-RL 5.4.0. The registered tasks are:

- `Mjlab-G1-Rickshaw-Flat-Teacher`
- `Mjlab-G1-Rickshaw-Flat-Student`
- the corresponding `-H91` history variants

## Physical Contract

The rickshaw wheel diameter is 0.6 m and each wheel center remains 0.3 m above
the ground plane. The rickshaw center of mass is shifted rearward by
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
The policy reference remains the physical pose `q_static`; the actuator layer
separately applies the static offset `tau / Kp` to its built-in position target.

Run the physical validations before training:

```bash
python scripts/validate_mjlab_assets.py
python scripts/validate_static_initialization.py
```

## Training And Playback

```bash
python scripts/train_teacher.py
python scripts/finetune_student.py --teacher <teacher.pt> --context <context.pt>
python scripts/play_student.py --checkpoint <student.pt>
```

The Mjlab runtime owns the flat plane, startup-fixed nine-parameter domain
randomization, online FAT2/ZMP diagnostics, observations, rewards, and RSL-RL
rollout state. Rewards use the Mjlab 1.5.3 G1 flat definitions directly, with
upright weight 0.2, foot swing target 0.08 m, and the two rickshaw hitch-height
terms. Commands are sampled directly every 3-8 s with 10% standing commands;
the tracked entity is the rickshaw. There is no secondary simulator runtime
path or runtime reward override.
