# G1 Rickshaw mjlab Commands

Install the project in the mjlab environment:

```bash
python -m pip install -e source/g1_rickshaw_lab
```

Validate the MuJoCo assets, fixed grippers, point connections, collision masks, and hitch geometry:

```bash
python scripts/validate_mjlab_assets.py
```

Validate and persist the flat-ground MuJoCo static equilibrium:

```bash
python scripts/validate_static_initialization.py
```

Train the teacher:

```bash
python scripts/train_teacher.py --num-envs 8192
```

Play or export the student:

```bash
python scripts/play_student.py --checkpoint <student-checkpoint.pt>
```

Render the solved initialization state:

```bash
python scripts/render_initialization.py --output outputs/initialization.png
```

Initialization loads the model-bound reset pose from
`config/static_rest_poses.json`. There is no asset conversion, gain ramp, or
settling controller.
