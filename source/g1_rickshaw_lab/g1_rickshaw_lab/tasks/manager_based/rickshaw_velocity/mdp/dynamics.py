"""Pure Torch rickshaw dynamics kernels."""

from __future__ import annotations

from dataclasses import MISSING, dataclass

import torch

GRAVITY = 9.81


def rolling_resistance_force(
    wheel_velocity_w: torch.Tensor,
    wheel_contact_force_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
    c_rr: torch.Tensor,
    *,
    velocity_epsilon: float = 0.05,
) -> torch.Tensor:
    """Compute wheel-center rolling resistance from the current contact force."""

    if wheel_velocity_w.ndim != 3 or wheel_velocity_w.shape[-1] != 3:
        raise ValueError("wheel_velocity_w must have shape [N, W, 3]")
    if wheel_contact_force_w.shape != wheel_velocity_w.shape:
        raise ValueError("wheel contact forces must match wheel velocity shape")
    if path_tangent_w.shape != (wheel_velocity_w.shape[0], 3):
        raise ValueError("path_tangent_w must have shape [N, 3]")
    if path_normal_w.shape != path_tangent_w.shape:
        raise ValueError("path normal and tangent shapes differ")
    if velocity_epsilon <= 0.0:
        raise ValueError("velocity_epsilon must be positive")

    tangential_velocity = torch.sum(wheel_velocity_w * path_tangent_w[:, None, :], dim=-1)
    normal_force = torch.clamp(
        torch.sum(wheel_contact_force_w * path_normal_w[:, None, :], dim=-1),
        min=0.0,
    )
    coefficient = torch.as_tensor(c_rr, device=wheel_velocity_w.device, dtype=wheel_velocity_w.dtype)
    if coefficient.ndim == 0:
        coefficient = coefficient.expand(wheel_velocity_w.shape[0])
    if coefficient.shape != (wheel_velocity_w.shape[0],):
        raise ValueError("c_rr must be scalar or have shape [N]")
    direction = torch.tanh(tangential_velocity / velocity_epsilon)
    magnitude = -coefficient[:, None] * normal_force * direction
    return magnitude[..., None] * path_tangent_w[:, None, :]


def wheel_longitudinal_slip(
    wheel_center_velocity_w: torch.Tensor,
    wheel_angular_velocity_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
    wheel_radius: float,
) -> torch.Tensor:
    """Return each wheel's longitudinal ground-contact velocity."""

    contact_offset_w = -wheel_radius * path_normal_w[:, None, :]
    contact_velocity_w = wheel_center_velocity_w + torch.cross(
        wheel_angular_velocity_w,
        contact_offset_w.expand_as(wheel_angular_velocity_w),
        dim=-1,
    )
    return torch.sum(contact_velocity_w * path_tangent_w[:, None, :], dim=-1)


@dataclass(frozen=True)
class RickshawMassProperties:
    """Per-environment cart quantities in the cart frame about the wheel axle."""

    m_cart: torch.Tensor
    com_x_from_axle: torch.Tensor
    com_z_from_axle: torch.Tensor
    pitch_inertia_about_axle: torch.Tensor
    m_eff: torch.Tensor
    b_eff: torch.Tensor
    handle_x_from_axle: torch.Tensor
    handle_z_from_axle: torch.Tensor


def parallel_axis_inertia(inertia_at_com: torch.Tensor, mass: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Shift a 3-D inertia tensor from its CoM by ``displacement``."""

    if inertia_at_com.shape[-2:] != (3, 3) or displacement.shape[-1] != 3:
        raise ValueError("inertia must end in [3,3] and displacement in [3]")
    eye = torch.eye(3, device=inertia_at_com.device, dtype=inertia_at_com.dtype)
    squared_distance = torch.sum(displacement * displacement, dim=-1)
    outer = displacement[..., :, None] * displacement[..., None, :]
    return inertia_at_com + mass[..., None, None] * (squared_distance[..., None, None] * eye - outer)


def combine_mass_properties(
    base_mass: torch.Tensor,
    base_com: torch.Tensor,
    base_inertia_at_com: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_com: torch.Tensor,
    payload_inertia_at_com: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine base and payload mass/CoM/inertia using parallel-axis shifts."""

    total_mass = base_mass + payload_mass
    if torch.any(total_mass <= 0.0):
        raise ValueError("combined mass must be positive")
    total_com = (base_mass[..., None] * base_com + payload_mass[..., None] * payload_com) / total_mass[..., None]
    base_shift = base_com - total_com
    payload_shift = payload_com - total_com
    total_inertia = parallel_axis_inertia(base_inertia_at_com, base_mass, base_shift) + parallel_axis_inertia(
        payload_inertia_at_com, payload_mass, payload_shift
    )
    return total_mass, total_com, total_inertia


def effective_cart_mass(
    cart_mass: torch.Tensor, wheel_spin_inertia: torch.Tensor, wheel_radius: torch.Tensor
) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return cart_mass + torch.sum(wheel_spin_inertia / torch.square(wheel_radius), dim=-1)


def effective_wheel_damping(wheel_damping: torch.Tensor, wheel_radius: torch.Tensor) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return torch.sum(wheel_damping / torch.square(wheel_radius), dim=-1)


@dataclass
class AnalyticForceCfg:
    minimum_wheel_normal_force: float = MISSING
    velocity_epsilon: float = 0.05
    minimum_handle_x: float = 0.5


@dataclass
class RickshawKinematicState:
    previous_velocity: torch.Tensor
    previous_angular_velocity_w: torch.Tensor
    forward_acceleration: torch.Tensor
    pitch_angular_velocity: torch.Tensor
    pitch_angular_acceleration: torch.Tensor
    yaw_angular_acceleration: torch.Tensor

    @classmethod
    def initialized(
        cls,
        tangential_velocity: torch.Tensor,
        angular_velocity_w: torch.Tensor,
    ) -> RickshawKinematicState:
        zeros = torch.zeros_like(tangential_velocity)
        return cls(
            previous_velocity=tangential_velocity.clone(),
            previous_angular_velocity_w=angular_velocity_w.clone(),
            forward_acceleration=zeros.clone(),
            pitch_angular_velocity=zeros.clone(),
            pitch_angular_acceleration=zeros.clone(),
            yaw_angular_acceleration=zeros.clone(),
        )

    def reset(
        self,
        tangential_velocity: torch.Tensor,
        angular_velocity_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.previous_velocity[ids] = tangential_velocity
        self.previous_angular_velocity_w[ids] = angular_velocity_w
        self.forward_acceleration[ids] = 0.0
        self.pitch_angular_velocity[ids] = 0.0
        self.pitch_angular_acceleration[ids] = 0.0
        self.yaw_angular_acceleration[ids] = 0.0

    def update(
        self,
        tangential_velocity: torch.Tensor,
        angular_velocity_w: torch.Tensor,
        path_lateral_w: torch.Tensor,
        path_normal_w: torch.Tensor,
        dt: float,
    ) -> None:
        angular_acceleration_w = (angular_velocity_w - self.previous_angular_velocity_w) / dt
        self.forward_acceleration[:] = (tangential_velocity - self.previous_velocity) / dt
        self.pitch_angular_velocity[:] = torch.sum(angular_velocity_w * path_lateral_w, dim=-1)
        self.pitch_angular_acceleration[:] = torch.sum(angular_acceleration_w * path_lateral_w, dim=-1)
        self.yaw_angular_acceleration[:] = torch.sum(angular_acceleration_w * path_normal_w, dim=-1)
        self.previous_velocity[:] = tangential_velocity
        self.previous_angular_velocity_w[:] = angular_velocity_w


@dataclass
class AnalyticHandleForceState:
    t_s: torch.Tensor
    t_n: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def initialized(cls, reference: torch.Tensor) -> AnalyticHandleForceState:
        zeros = torch.zeros_like(reference)
        return cls(
            t_s=zeros.clone(),
            t_n=zeros.clone(),
            valid=torch.zeros_like(reference, dtype=torch.bool),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.t_s[ids] = 0.0
        self.t_n[ids] = 0.0
        self.valid[ids] = False


def analytic_handle_force(
    v_s: torch.Tensor,
    a_s: torch.Tensor,
    alpha_ddot: torch.Tensor,
    alpha: torch.Tensor,
    c_rr: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    properties: RickshawMassProperties,
    *,
    velocity_epsilon: float = 0.05,
    minimum_handle_x: float = 0.5,
    handle_from_axle_sn: torch.Tensor | None = None,
    com_from_axle_sn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the complete cart tangent force and axle moment balance.

    The stored CoM and handle coordinates are cart-frame vectors from the axle.
    They are rotated by the current front-lift pitch before evaluating moments
    in the ground-aligned path frame.
    """

    if velocity_epsilon <= 0.0:
        raise ValueError("velocity_epsilon must be positive")
    n_w = torch.sum(wheel_normal_force, dim=-1)
    rr_magnitude = c_rr * n_w * torch.tanh(v_s / velocity_epsilon)
    t_s = properties.m_eff * a_s + rr_magnitude + properties.b_eff * v_s
    if (handle_from_axle_sn is None) != (com_from_axle_sn is None):
        raise ValueError("actual handle and CoM geometry must be supplied together")
    if handle_from_axle_sn is None:
        cosine = torch.cos(alpha)
        sine = torch.sin(alpha)
        handle_x = cosine * properties.handle_x_from_axle - sine * properties.handle_z_from_axle
        handle_z = sine * properties.handle_x_from_axle + cosine * properties.handle_z_from_axle
        com_x = cosine * properties.com_x_from_axle - sine * properties.com_z_from_axle
        com_z = sine * properties.com_x_from_axle + cosine * properties.com_z_from_axle
    else:
        expected_shape = (v_s.shape[0], 2)
        if handle_from_axle_sn.shape != expected_shape or com_from_axle_sn.shape != expected_shape:
            raise ValueError(f"actual cart geometry must have shape {expected_shape}")
        handle_x, handle_z = handle_from_axle_sn.unbind(dim=-1)
        com_x, com_z = com_from_axle_sn.unbind(dim=-1)
    valid = handle_x > minimum_handle_x
    denominator = torch.where(valid, handle_x, torch.ones_like(v_s))
    t_n = (
        properties.pitch_inertia_about_axle * alpha_ddot + handle_z * t_s + properties.m_cart * GRAVITY * com_x
    ) / denominator
    t_s = torch.where(valid, t_s, torch.zeros_like(t_s))
    t_n = torch.where(valid, t_n, torch.zeros_like(t_n))
    return t_s, t_n, valid


def update_analytic_handle_force_state(
    state: AnalyticHandleForceState,
    v_s: torch.Tensor,
    a_s: torch.Tensor,
    pitch: torch.Tensor,
    pitch_angular_acceleration: torch.Tensor,
    c_rr: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    properties: RickshawMassProperties,
    cfg: AnalyticForceCfg,
    *,
    handle_from_axle_sn: torch.Tensor | None = None,
    com_from_axle_sn: torch.Tensor | None = None,
) -> AnalyticHandleForceState:
    """Update the analytic handle force from the current cart kinematics."""

    t_s, t_n, geometry_valid = analytic_handle_force(
        v_s,
        a_s,
        pitch_angular_acceleration,
        pitch,
        c_rr,
        wheel_normal_force,
        properties,
        velocity_epsilon=cfg.velocity_epsilon,
        minimum_handle_x=cfg.minimum_handle_x,
        handle_from_axle_sn=handle_from_axle_sn,
        com_from_axle_sn=com_from_axle_sn,
    )
    wheel_valid = torch.all(wheel_normal_force >= cfg.minimum_wheel_normal_force, dim=-1)
    state.t_s[:] = t_s
    state.t_n[:] = t_n
    state.valid[:] = geometry_valid & wheel_valid
    return state


def quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions without a simulator dependency."""

    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector dimensions must end in 4/3")
    q_vec = quaternion[..., 1:]
    uv = torch.linalg.cross(q_vec, vector, dim=-1)
    uuv = torch.linalg.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def wheel_ground_frame(
    quaternion_wxyz: torch.Tensor,
    ground_normal_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forward, axle-lateral, and normal axes for a wheeled body."""

    axle_b = torch.zeros_like(ground_normal_w)
    axle_b[..., 1] = 1.0
    axle_w = quat_apply_wxyz(quaternion_wxyz, axle_b)
    forward_w = torch.nn.functional.normalize(torch.cross(axle_w, ground_normal_w, dim=-1), dim=-1)
    lateral_w = torch.cross(ground_normal_w, forward_w, dim=-1)
    return forward_w, lateral_w, ground_normal_w


def yaw_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return world yaw from a wxyz quaternion."""

    local_x = torch.zeros((*quaternion.shape[:-1], 3), device=quaternion.device, dtype=quaternion.dtype)
    local_x[..., 0] = 1.0
    forward_w = quat_apply_wxyz(quaternion, local_x)
    return torch.atan2(forward_w[..., 1], forward_w[..., 0])


def relative_position_in_yaw_frame(
    origin_position_w: torch.Tensor,
    point_position_w: torch.Tensor,
    frame_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Express a relative position in the frame's yaw-only coordinates."""

    relative_w = point_position_w - origin_position_w
    yaw = yaw_from_quaternion_wxyz(frame_quaternion_wxyz)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * relative_w[..., 0] + sine * relative_w[..., 1],
            -sine * relative_w[..., 0] + cosine * relative_w[..., 1],
            relative_w[..., 2],
        ),
        dim=-1,
    )


def relative_yaw_from_quaternions(
    reference_quaternion_wxyz: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return target yaw relative to the reference frame in [-pi, pi]."""

    difference = yaw_from_quaternion_wxyz(target_quaternion_wxyz) - yaw_from_quaternion_wxyz(reference_quaternion_wxyz)
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def connect_constraint_forces(
    efc_type: torch.Tensor,
    efc_id: torch.Tensor,
    efc_force: torch.Tensor,
    equality_ids: torch.Tensor,
    *,
    equality_constraint_type: int,
) -> torch.Tensor:
    """Extract the world xyz solve force for each site-connect equality."""

    equality_row_count = 3 * equality_ids.numel()
    matches = (efc_type[:, :equality_row_count, None] == equality_constraint_type) & (
        efc_id[:, :equality_row_count, None] == equality_ids[None, None, :]
    )
    first_row = torch.argmax(matches.to(torch.int64), dim=1)
    xyz_rows = first_row[:, :, None] + torch.arange(3, device=efc_force.device)
    return torch.gather(efc_force, 1, xyz_rows.reshape(efc_force.shape[0], -1)).reshape(
        efc_force.shape[0], equality_ids.numel(), 3
    )


def rickshaw_pitch_from_quaternion(
    quaternion_wxyz: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Return front-lift pitch ``alpha`` relative to the path frame."""

    local_x = torch.zeros_like(path_tangent_w)
    local_x[..., 0] = 1.0
    forward_w = quat_apply_wxyz(quaternion_wxyz, local_x)
    forward_s = torch.sum(forward_w * path_tangent_w, dim=-1)
    forward_n = torch.sum(forward_w * path_normal_w, dim=-1)
    return torch.atan2(forward_n, forward_s)


__all__ = [
    "AnalyticForceCfg",
    "AnalyticHandleForceState",
    "GRAVITY",
    "RickshawMassProperties",
    "RickshawKinematicState",
    "analytic_handle_force",
    "combine_mass_properties",
    "connect_constraint_forces",
    "effective_cart_mass",
    "effective_wheel_damping",
    "parallel_axis_inertia",
    "quat_apply_wxyz",
    "relative_position_in_yaw_frame",
    "relative_yaw_from_quaternions",
    "rickshaw_pitch_from_quaternion",
    "rolling_resistance_force",
    "update_analytic_handle_force_state",
    "wheel_longitudinal_slip",
    "wheel_ground_frame",
    "yaw_from_quaternion_wxyz",
]
