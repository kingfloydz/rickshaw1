"""Joint-space reference motion used by the optional mimic command."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

LEG_TORSO_JOINT_COUNT = 15


def joint_velocity_from_positions(
    joint_pos: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    joint_vel = torch.empty_like(joint_pos)
    joint_vel[0] = (joint_pos[1] - joint_pos[0]) * fps
    joint_vel[-1] = (joint_pos[-1] - joint_pos[-2]) * fps
    joint_vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) * (0.5 * fps)
    return joint_vel


@dataclass(frozen=True, slots=True)
class JointMotionReference:
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    @property
    def duration_s(self) -> float:
        return self.joint_pos.shape[0] / self.fps

    def duration_steps(self, step_dt: float) -> int:
        return round(self.duration_s / step_dt)

    def sample(self, elapsed_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame = elapsed_s * self.fps
        lower = torch.floor(frame).to(torch.long).clamp(max=self.joint_pos.shape[0] - 1)
        upper = (lower + 1).clamp(max=self.joint_pos.shape[0] - 1)
        alpha = (frame - lower).clamp(0.0, 1.0).unsqueeze(-1)
        joint_pos = torch.lerp(self.joint_pos[lower], self.joint_pos[upper], alpha)
        joint_vel = torch.lerp(self.joint_vel[lower], self.joint_vel[upper], alpha)
        return joint_pos, joint_vel


def load_joint_motion_reference(
    path: str | Path,
    device: torch.device | str,
) -> JointMotionReference:
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    fps = float(data["fps"])
    joint_pos = torch.as_tensor(
        data["dof_pos"][:, :LEG_TORSO_JOINT_COUNT],
        dtype=torch.float32,
        device=device,
    )
    return JointMotionReference(
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_velocity_from_positions(joint_pos, fps),
    )


__all__ = [
    "LEG_TORSO_JOINT_COUNT",
    "JointMotionReference",
    "joint_velocity_from_positions",
    "load_joint_motion_reference",
]
