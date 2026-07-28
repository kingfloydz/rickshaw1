from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import train_pipeline


def test_pipeline_passes_latent_dim_and_checkpoints_between_stages(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == train_pipeline.PROJECT_ROOT
        assert check is True
        commands.append(command)
        experiment_dir = Path(command[command.index("--experiment-dir") + 1])
        run_dir = experiment_dir / f"run_{len(commands)}"
        run_dir.mkdir()
        (run_dir / "model_50.pt").touch()
        (run_dir / "model_100.pt").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(train_pipeline.subprocess, "run", fake_run)
    args = train_pipeline.PipelineArgs(latent_dim=8, log_root=tmp_path)
    old_run = tmp_path / "g1_rickshaw_teacher" / "run_10"
    old_run.mkdir(parents=True)
    (old_run / "model_9999.pt").touch()

    teacher, context, student = train_pipeline.run_pipeline(args)

    assert [command[1] for command in commands] == [
        str(train_pipeline.SCRIPTS_DIR / "train_teacher.py"),
        str(train_pipeline.SCRIPTS_DIR / "train_context.py"),
        str(train_pipeline.SCRIPTS_DIR / "finetune_student.py"),
    ]
    assert all(
        command[command.index("--latent-dim") + 1] == "8" for command in commands
    )
    assert commands[1][commands[1].index("--teacher") + 1] == str(teacher)
    assert commands[2][commands[2].index("--teacher") + 1] == str(teacher)
    assert commands[2][commands[2].index("--context") + 1] == str(context)
    assert teacher.name == context.name == student.name == "model_100.pt"


def test_pipeline_uses_stage_iteration_defaults() -> None:
    args = train_pipeline.PipelineArgs()
    experiment_dir = Path("logs/rsl_rl/g1_rickshaw_teacher")

    command = train_pipeline._stage_command(
        "train_teacher.py", args, experiment_dir, args.s0_iterations
    )

    assert command[command.index("--max-iterations") + 1] == "6000"
    assert args.s1_iterations == 6_000
    assert args.s2_iterations == 800
