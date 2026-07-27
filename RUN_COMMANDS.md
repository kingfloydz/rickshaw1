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



python scripts/play.py \
  Mjlab-G1-Rickshaw-Slopes-Teacher \
  --checkpoint-file logs/rsl_rl/g1_rickshaw_teacher/2026-07-26_16-23-24_s0/model_4950.pt \
  --num-envs 19 \
  --device cuda:0 \
  --viewer viser

<<<<<<< Updated upstream
python scripts/train_context.py \
  --task Mjlab-G1-Rickshaw-Slopes-Teacher \
  --teacher "$TEACHER" \
  --output "$CONTEXT" \
  --num-envs 8192 \
  --device cuda:0 \
  --max-iterations 2000

python scripts/finetune_student.py \
  --task Mjlab-G1-Rickshaw-Slopes-Student \
  --teacher "$TEACHER" \
  --context "$CONTEXT" \
  --num-envs 8192 \
  --device cuda:0
=======

cd /inspire/hdd/project/leverage-robot/ky26212/slopes

PYTHONPATH="$PWD/source/g1_rickshaw_lab${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/play.py Mjlab-G1-Rickshaw-Slopes-Teacher \
  --checkpoint-file /inspire/hdd/project/leverage-robot/ky26212/slopes/logs/rsl_rl/g1_rickshaw_teacher/2026-07-26_16-23-24_s0/model_4950.pt \
  --slope 0.10 \
  --num-envs 1 \
  --device cuda:0 \
  --viewer viser
>>>>>>> Stashed changes
