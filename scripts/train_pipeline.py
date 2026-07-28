#!/usr/bin/env python3
"""Run S0 teacher training, S1 distillation, and S2 fine-tuning in sequence."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


@dataclass
class PipelineArgs:
    latent_dim: int = 16
    history_length: int = 61
    rollout_steps: int = 24
    num_envs: int = 8192
    s0_iterations: int = 6_000
    s1_iterations: int = 6_000
    s2_iterations: int = 800
    seed: int = 42
    log_root: Path = Path("logs/rsl_rl")
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def _gpu_cli(gpu_ids: list[int] | Literal["all"] | None) -> list[str]:
    if gpu_ids is None:
        return ["--gpu-ids", "None"]
    if gpu_ids == "all":
        return ["--gpu-ids", "all"]
    return ["--gpu-ids", *(str(gpu_id) for gpu_id in gpu_ids)]


def _stage_command(
    script: str,
    args: PipelineArgs,
    experiment_dir: Path,
    max_iterations: int,
    *checkpoint_args: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS_DIR / script),
        *checkpoint_args,
        "--experiment-dir",
        str(experiment_dir),
        "--latent-dim",
        str(args.latent_dim),
        "--history-length",
        str(args.history_length),
        "--rollout-steps",
        str(args.rollout_steps),
        "--num-envs",
        str(args.num_envs),
        "--max-iterations",
        str(max_iterations),
        "--seed",
        str(args.seed),
        *_gpu_cli(args.gpu_ids),
    ]


def _run_stage(label: str, command: list[str], experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = {path.name for path in experiment_dir.iterdir() if path.is_dir()}

    print(f"[PIPELINE] Starting {label}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    new_runs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and path.name not in existing_runs
    ]
    if len(new_runs) != 1:
        raise RuntimeError(
            f"{label} created {len(new_runs)} run directories in {experiment_dir}"
        )

    from mjlab.utils.os import get_checkpoint_path

    run_dir = new_runs[0]
    checkpoint = get_checkpoint_path(
        experiment_dir,
        run_dir=rf"^{re.escape(run_dir.name)}$",
        checkpoint=r"^model_\d+\.pt$",
    ).resolve()
    print(f"[PIPELINE] Finished {label}: {checkpoint}", flush=True)
    return checkpoint


def run_pipeline(args: PipelineArgs) -> tuple[Path, Path, Path]:
    log_root = args.log_root.resolve()
    teacher_dir = log_root / "g1_rickshaw_teacher"
    context_dir = log_root / "g1_rickshaw_context"
    student_dir = log_root / "g1_rickshaw_student"

    teacher = _run_stage(
        "S0 teacher training",
        _stage_command("train_teacher.py", args, teacher_dir, args.s0_iterations),
        teacher_dir,
    )
    context = _run_stage(
        "S1 online distillation",
        _stage_command(
            "train_context.py",
            args,
            context_dir,
            args.s1_iterations,
            "--teacher",
            str(teacher),
        ),
        context_dir,
    )
    student = _run_stage(
        "S2 student fine-tuning",
        _stage_command(
            "finetune_student.py",
            args,
            student_dir,
            args.s2_iterations,
            "--teacher",
            str(teacher),
            "--context",
            str(context),
        ),
        student_dir,
    )
    return teacher, context, student


def main() -> None:
    teacher, context, student = run_pipeline(tyro.cli(PipelineArgs))
    print("[PIPELINE] All stages completed", flush=True)
    print(f"S0 checkpoint: {teacher}")
    print(f"S1 checkpoint: {context}")
    print(f"S2 checkpoint: {student}")


if __name__ == "__main__":
    main()
