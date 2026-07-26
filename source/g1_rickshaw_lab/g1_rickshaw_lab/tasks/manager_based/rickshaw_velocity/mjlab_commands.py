"""Rickshaw velocity command in the wheel/ground-aligned frame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from .mdp.dynamics import wheel_ground_frame
from .mdp.mimic import JointMotionReference, load_joint_motion_reference


def rickshaw_frame(asset: Any, slope_normal_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return wheel_ground_frame(asset.data.root_link_quat_w, slope_normal_w)


def rickshaw_velocity(asset: Any, slope_normal_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_w, lateral_w, normal_w = rickshaw_frame(asset, slope_normal_w)
    linear = asset.data.root_link_lin_vel_w
    return (
        torch.sum(linear * forward_w, dim=-1),
        torch.sum(linear * lateral_w, dim=-1),
        torch.sum(asset.data.root_link_ang_vel_w * normal_w, dim=-1),
    )


@dataclass(kw_only=True)
class RickshawVelocityCommandCfg(UniformVelocityCommandCfg):
    mimic: bool = False
    mimic_motion_path: str = ""
    mimic_forward_speed: float = 0.8

    def build(self, env):
        return RickshawVelocityCommand(self, env)


class RickshawVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg: RickshawVelocityCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.metrics = {
            "error_lin_vel_x": torch.zeros(self.num_envs, device=self.device),
            "error_ang_vel_z": torch.zeros(self.num_envs, device=self.device),
        }
        self._rickshaw_joystick_enabled = None
        self._rickshaw_joystick_sliders: dict[int, Any] = {}
        self.is_mimic_env = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.mimic_step = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.mimic_duration_steps = 0
        self.mimic_reference: JointMotionReference | None = None
        if cfg.mimic:
            reference = load_joint_motion_reference(
                cfg.mimic_motion_path,
                self.device,
            )
            self.mimic_reference = reference
            self.mimic_duration_steps = reference.duration_steps(env.step_dt)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.is_mimic_env[env_ids] = False
        self.mimic_step[env_ids] = 0
        if self.cfg.mimic:
            reference = cast(JointMotionReference, self.mimic_reference)
            mimic_env_ids = env_ids[self.is_forward_env[env_ids]]
            self.is_mimic_env[mimic_env_ids] = True
            self.is_standing_env[mimic_env_ids] = False
            self.is_heading_env[mimic_env_ids] = False
            self.is_world_env[mimic_env_ids] = False
            self.vel_command_b[mimic_env_ids, 0] = self.cfg.mimic_forward_speed
            self.vel_command_b[mimic_env_ids, 1:] = 0.0
            self.vel_command_w[mimic_env_ids] = 0.0
            self.time_left[mimic_env_ids] = reference.duration_s

    def sample_mimic_reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        reference = cast(JointMotionReference, self.mimic_reference)
        elapsed_s = self.mimic_step * self._env.step_dt
        return reference.sample(elapsed_s)

    def create_gui(
        self,
        name: str,
        server: Any,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Create controls for the two command axes used by the rickshaw task."""

        del on_change, request_action
        from viser import Icon

        axes = (
            (0, "lin_vel_x", self.cfg.ranges.lin_vel_x),
            (2, "ang_vel_z", self.cfg.ranges.ang_vel_z),
        )
        sliders: dict[int, Any] = {}
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for command_index, label, limits in axes:
                max_value = max(abs(float(limits[0])), abs(float(limits[1])))
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_value,
                    step=0.1,
                    min=0.1,
                    max=10.0,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_value,
                    max=max_value,
                    step=0.05,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _update_range(_event, target=slider, maximum=max_input) -> None:
                    target.min = -maximum.value
                    target.max = maximum.value

                sliders[command_index] = slider

            zero_button = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_button.on_click
            def _zero(_event) -> None:
                for slider in sliders.values():
                    slider.value = 0.0

        self._rickshaw_joystick_enabled = enabled
        self._rickshaw_joystick_sliders = sliders
        self._rickshaw_joystick_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        self.mimic_step[self.is_mimic_env] += int(dt > 0.0)
        expired = self.is_mimic_env & (
            self.mimic_step >= self.mimic_duration_steps
        )
        self.time_left[expired] = 0.0
        super().compute(dt)
        enabled = self._rickshaw_joystick_enabled
        if enabled is not None and enabled.value:
            env_idx = self._rickshaw_joystick_get_env_idx()
            self.vel_command_b[env_idx, 1] = 0.0
            for command_index, slider in self._rickshaw_joystick_sliders.items():
                self.vel_command_b[env_idx, command_index] = slider.value

    def _update_metrics(self) -> None:
        lin_vel_x, _, ang_vel_z = rickshaw_velocity(self.robot, self._env.path_normal_w)
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_lin_vel_x"] += torch.abs(self.vel_command_b[:, 0] - lin_vel_x) / max_command_step
        self.metrics["error_ang_vel_z"] += torch.abs(self.vel_command_b[:, 2] - ang_vel_z) / max_command_step

    def _debug_vis_impl(self, visualizer: Any) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        forward_w, lateral_w, normal_w = rickshaw_frame(self.robot, self._env.path_normal_w)
        command = self.command
        linear_velocity_w = self.robot.data.root_link_lin_vel_w
        angular_velocity_w = self.robot.data.root_link_ang_vel_w
        command_linear_w = command[:, :1] * forward_w + command[:, 1:2] * lateral_w
        actual_linear_w = (
            torch.sum(linear_velocity_w * forward_w, dim=-1, keepdim=True) * forward_w
            + torch.sum(linear_velocity_w * lateral_w, dim=-1, keepdim=True) * lateral_w
        )
        command_angular_w = command[:, 2:3] * normal_w
        actual_angular_w = torch.sum(angular_velocity_w * normal_w, dim=-1, keepdim=True) * normal_w
        start_w = self.robot.data.root_link_pos_w + (self.cfg.viz.scale * self.cfg.viz.z_offset * normal_w)
        scale = self.cfg.viz.scale
        for env_idx in env_indices:
            start = start_w[env_idx]
            visualizer.add_arrow(
                start,
                start + scale * command_linear_w[env_idx],
                color=(0.2, 0.2, 0.6, 0.6),
                width=0.015,
            )
            visualizer.add_arrow(
                start,
                start + scale * command_angular_w[env_idx],
                color=(0.2, 0.6, 0.2, 0.6),
                width=0.015,
            )
            visualizer.add_arrow(
                start,
                start + scale * actual_linear_w[env_idx],
                color=(0.0, 0.6, 1.0, 0.7),
                width=0.015,
            )
            visualizer.add_arrow(
                start,
                start + scale * actual_angular_w[env_idx],
                color=(0.0, 1.0, 0.4, 0.7),
                width=0.015,
            )


__all__ = [
    "RickshawVelocityCommand",
    "RickshawVelocityCommandCfg",
    "rickshaw_frame",
    "rickshaw_velocity",
]
