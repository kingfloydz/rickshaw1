"""Pure Torch cart, FAT2, and ZMP dynamics kernels."""

from __future__ import annotations

import math
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

    tangential_velocity = torch.sum(
        wheel_velocity_w * path_tangent_w[:, None, :], dim=-1
    )
    normal_force = torch.clamp(
        torch.sum(wheel_contact_force_w * path_normal_w[:, None, :], dim=-1),
        min=0.0,
    )
    coefficient = torch.as_tensor(
        c_rr, device=wheel_velocity_w.device, dtype=wheel_velocity_w.dtype
    )
    if coefficient.ndim == 0:
        coefficient = coefficient.expand(wheel_velocity_w.shape[0])
    if coefficient.shape != (wheel_velocity_w.shape[0],):
        raise ValueError("c_rr must be scalar or have shape [N]")
    direction = torch.tanh(tangential_velocity / velocity_epsilon)
    magnitude = -coefficient[:, None] * normal_force * direction
    return magnitude[..., None] * path_tangent_w[:, None, :]


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


def articulation_center_of_mass(
    body_com_pos_w: torch.Tensor,
    body_com_lin_vel_w: torch.Tensor,
    body_masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mass-weighted whole-articulation CoM position and velocity.

    Root-link ``root_com_*`` fields describe only the root rigid body. ZMP
    and FAT require the system CoM across every retained articulation body.
    """

    if body_com_pos_w.ndim != 3 or body_com_pos_w.shape[-1] != 3:
        raise ValueError("body CoM positions must have shape [N,B,3]")
    if body_com_lin_vel_w.shape != body_com_pos_w.shape:
        raise ValueError("body CoM linear velocities must match positions")
    if body_masses.shape != body_com_pos_w.shape[:2]:
        raise ValueError("body masses must have shape [N,B]")
    if torch.any(~torch.isfinite(body_masses)) or torch.any(body_masses <= 0.0):
        raise ValueError("every retained articulation body must have finite positive mass")
    total_mass = torch.sum(body_masses, dim=-1)
    weights = body_masses / total_mass[:, None]
    position = torch.sum(body_com_pos_w * weights[..., None], dim=1)
    velocity = torch.sum(body_com_lin_vel_w * weights[..., None], dim=1)
    return position, velocity, total_mass


def parallel_axis_inertia(
    inertia_at_com: torch.Tensor, mass: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """Shift a 3-D inertia tensor from its CoM by ``displacement``."""

    if inertia_at_com.shape[-2:] != (3, 3) or displacement.shape[-1] != 3:
        raise ValueError("inertia must end in [3,3] and displacement in [3]")
    eye = torch.eye(3, device=inertia_at_com.device, dtype=inertia_at_com.dtype)
    squared_distance = torch.sum(displacement * displacement, dim=-1)
    outer = displacement[..., :, None] * displacement[..., None, :]
    return inertia_at_com + mass[..., None, None] * (
        squared_distance[..., None, None] * eye - outer
    )


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
    total_com = (
        base_mass[..., None] * base_com + payload_mass[..., None] * payload_com
    ) / total_mass[..., None]
    base_shift = base_com - total_com
    payload_shift = payload_com - total_com
    total_inertia = parallel_axis_inertia(
        base_inertia_at_com, base_mass, base_shift
    ) + parallel_axis_inertia(payload_inertia_at_com, payload_mass, payload_shift)
    return total_mass, total_com, total_inertia


def effective_cart_mass(
    cart_mass: torch.Tensor, wheel_spin_inertia: torch.Tensor, wheel_radius: torch.Tensor
) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return cart_mass + torch.sum(wheel_spin_inertia / torch.square(wheel_radius), dim=-1)


def effective_wheel_damping(
    wheel_damping: torch.Tensor, wheel_radius: torch.Tensor
) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return torch.sum(wheel_damping / torch.square(wheel_radius), dim=-1)


@dataclass
class AnalyticForceCfg:
    minimum_wheel_normal_force: float = MISSING
    velocity_epsilon: float = 0.05
    minimum_handle_x: float = 0.5


@dataclass
class AnalyticHandleForceState:
    previous_velocity: torch.Tensor
    previous_pitch: torch.Tensor
    previous_previous_pitch: torch.Tensor
    a_s: torch.Tensor
    alpha_ddot: torch.Tensor
    t_s: torch.Tensor
    t_n: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def initialized(
        cls, tangential_velocity: torch.Tensor, pitch: torch.Tensor
    ) -> AnalyticHandleForceState:
        zeros = torch.zeros_like(tangential_velocity)
        return cls(
            previous_velocity=tangential_velocity.clone(),
            previous_pitch=pitch.clone(),
            previous_previous_pitch=pitch.clone(),
            a_s=zeros.clone(),
            alpha_ddot=zeros.clone(),
            t_s=zeros.clone(),
            t_n=zeros.clone(),
            valid=torch.zeros_like(tangential_velocity, dtype=torch.bool),
        )

    def reset(
        self,
        tangential_velocity: torch.Tensor,
        pitch: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.previous_velocity[ids] = tangential_velocity
        self.previous_pitch[ids] = pitch
        self.previous_previous_pitch[ids] = pitch
        self.a_s[ids] = 0.0
        self.alpha_ddot[ids] = 0.0
        self.t_s[ids] = 0.0
        self.t_n[ids] = 0.0
        self.valid[ids] = False


def force_consistency(
    analytic_force_sn: torch.Tensor,
    measured_force_sn: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_floor_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare current analytic and measured handle forces for FAT2 validity."""

    if analytic_force_sn.ndim != 2 or analytic_force_sn.shape[1] != 2:
        raise ValueError("analytic handle force must have shape [N, 2]")
    if measured_force_sn.shape != analytic_force_sn.shape:
        raise ValueError("measured handle force must match analytic handle force")
    if source_valid.shape != analytic_force_sn.shape[:1] or source_valid.dtype != torch.bool:
        raise ValueError("force consistency source_valid must be bool [N]")
    if not 0.0 <= relative_tolerance <= 1.0 or absolute_floor_n <= 0.0:
        raise ValueError("force consistency tolerances are invalid")
    floor = torch.as_tensor(
        absolute_floor_n, device=analytic_force_sn.device, dtype=analytic_force_sn.dtype
    )
    normalization_force = torch.maximum(torch.abs(analytic_force_sn), floor)
    relative_error = torch.abs(measured_force_sn - analytic_force_sn) / normalization_force
    sign_resolved = (
        (torch.abs(analytic_force_sn) > relative_tolerance * normalization_force)
        & (torch.abs(measured_force_sn) > relative_tolerance * normalization_force)
    )
    same_sign = (~sign_resolved) | (
        torch.sign(analytic_force_sn) == torch.sign(measured_force_sn)
    )
    consistent = source_valid & torch.all(
        same_sign & (relative_error <= relative_tolerance), dim=-1
    )
    return consistent, relative_error


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
    in the flat-ground path frame.
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
        properties.pitch_inertia_about_axle * alpha_ddot
        + handle_z * t_s
        + properties.m_cart * GRAVITY * com_x
    ) / denominator
    t_s = torch.where(valid, t_s, torch.zeros_like(t_s))
    t_n = torch.where(valid, t_n, torch.zeros_like(t_n))
    return t_s, t_n, valid


def update_analytic_handle_force_state(
    state: AnalyticHandleForceState,
    v_s: torch.Tensor,
    pitch: torch.Tensor,
    c_rr: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    properties: RickshawMassProperties,
    dt: float,
    cfg: AnalyticForceCfg,
    *,
    handle_from_axle_sn: torch.Tensor | None = None,
    com_from_axle_sn: torch.Tensor | None = None,
) -> AnalyticHandleForceState:
    """Differentiate cart motion and update the analytic FAT2 reference."""

    a_s = (v_s - state.previous_velocity) / dt
    alpha_ddot = (
        pitch - 2.0 * state.previous_pitch + state.previous_previous_pitch
    ) / (dt * dt)
    state.previous_velocity[:] = v_s
    state.previous_previous_pitch[:] = state.previous_pitch
    state.previous_pitch[:] = pitch
    t_s, t_n, geometry_valid = analytic_handle_force(
        v_s,
        a_s,
        alpha_ddot,
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
    state.a_s[:] = a_s
    state.alpha_ddot[:] = alpha_ddot
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


def torso_pitch_from_world_vertical(
    torso_quaternion_wxyz: torch.Tensor,
    path_tangent_w: torch.Tensor,
) -> torch.Tensor:
    """Return torso tilt from world vertical, positive along the path."""

    local_z = torch.zeros_like(path_tangent_w)
    local_z[..., 2] = 1.0
    up_w = quat_apply_wxyz(torso_quaternion_wxyz, local_z)
    world_up = torch.zeros_like(path_tangent_w)
    world_up[..., 2] = 1.0
    horizontal_forward = path_tangent_w - torch.sum(
        path_tangent_w * world_up, dim=-1, keepdim=True
    ) * world_up
    horizontal_norm = torch.linalg.vector_norm(horizontal_forward, dim=-1, keepdim=True)
    if torch.any(horizontal_norm <= 1.0e-6):
        raise ValueError("path tangent must have a nonzero horizontal projection")
    horizontal_forward = horizontal_forward / horizontal_norm
    return torch.atan2(
        torch.sum(up_w * horizontal_forward, dim=-1),
        torch.sum(up_w * world_up, dim=-1),
    )


@dataclass
class FAT2Cfg:
    robot_mass: float = MISSING
    com_radius: float = MISSING
    com_radius_bounds: tuple[float, float] = MISSING
    theta_max: float = MISSING
    force_consistency_relative_tolerance: float = MISSING
    force_consistency_absolute_floor_n: float = MISSING

    def validate(self) -> None:
        if self.robot_mass <= 0.0 or self.com_radius <= 0.0:
            raise ValueError("FAT2 robot mass and CoM radius must be calibrated")
        if len(self.com_radius_bounds) != 2:
            raise ValueError("FAT2 CoM radius bounds must contain two values")
        radius_min, radius_max = self.com_radius_bounds
        if radius_min <= 0.0 or radius_min >= radius_max:
            raise ValueError("FAT2 CoM radius bounds must be positive and ordered")
        if not radius_min <= self.com_radius <= radius_max:
            raise ValueError("FAT2 calibrated CoM radius must lie within its bounds")
        if not 0.0 < self.theta_max < math.pi / 2.0:
            raise ValueError("FAT2 theta_max must lie in (0, pi/2)")
        if not 0.0 <= self.force_consistency_relative_tolerance <= 1.0:
            raise ValueError("FAT2 force relative tolerance must lie in [0,1]")
        if self.force_consistency_absolute_floor_n <= 0.0:
            raise ValueError("FAT2 force absolute floor must be positive")


def fat2_reference_angle(
    handle_s: torch.Tensor,
    handle_n: torch.Tensor,
    hand_force_s: torch.Tensor,
    hand_force_n: torch.Tensor,
    robot_mass: torch.Tensor | float,
    com_radius: torch.Tensor | float,
    theta_max: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the hand-force FAT2 weak torso prior."""

    mass = torch.as_tensor(robot_mass, device=handle_s.device, dtype=handle_s.dtype)
    radius = torch.as_tensor(com_radius, device=handle_s.device, dtype=handle_s.dtype)
    maximum = torch.as_tensor(theta_max, device=handle_s.device, dtype=handle_s.dtype)
    if torch.any(mass <= 0.0) or torch.any(radius <= 0.0):
        raise ValueError("robot_mass and com_radius must be positive")
    if torch.any((maximum <= 0.0) | (maximum >= math.pi / 2.0)):
        raise ValueError("theta_max must lie in (0, pi/2)")
    hand_moment = handle_s * hand_force_n - handle_n * hand_force_s
    ratio = hand_moment / (mass * GRAVITY * radius)
    limit = torch.sin(maximum)
    return torch.asin(torch.clamp(ratio, min=-limit, max=limit))


def sagittal_com_radius(
    robot_com_w: torch.Tensor,
    support_center_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Return support-to-CoM distance in the path tangent/normal plane."""

    if robot_com_w.ndim != 2 or robot_com_w.shape[-1] != 3:
        raise ValueError("robot CoM must have shape [N,3]")
    if any(
        value.shape != robot_com_w.shape
        for value in (support_center_w, path_tangent_w, path_normal_w)
    ):
        raise ValueError("FAT2 sagittal geometry tensors must have identical shapes")
    offset = robot_com_w - support_center_w
    offset_s = torch.sum(offset * path_tangent_w, dim=-1)
    offset_n = torch.sum(offset * path_normal_w, dim=-1)
    return torch.sqrt(torch.square(offset_s) + torch.square(offset_n))


@dataclass
class ZMPCfg:
    min_ground_reaction: float = MISSING


@dataclass(kw_only=True)
class SupportPolygonCfg:
    foot_half_length: float = MISSING
    foot_half_width: float = MISSING
    foot_center_offset_x: float = MISSING

    def validate(self) -> None:
        if self.foot_half_length <= 0.0 or self.foot_half_width <= 0.0:
            raise ValueError("calibrated foot half dimensions must be positive")
        if not math.isfinite(self.foot_center_offset_x):
            raise ValueError("calibrated foot center offset must be finite")


@dataclass
class ZMPKinematicState:
    previous_velocity_s: torch.Tensor
    previous_velocity_n: torch.Tensor
    acceleration_s: torch.Tensor
    acceleration_n: torch.Tensor

    @classmethod
    def initialized(
        cls, velocity_s: torch.Tensor, velocity_n: torch.Tensor
    ) -> ZMPKinematicState:
        zeros = torch.zeros_like(velocity_s)
        return cls(
            previous_velocity_s=velocity_s.clone(),
            previous_velocity_n=velocity_n.clone(),
            acceleration_s=zeros.clone(),
            acceleration_n=zeros.clone(),
        )

    def update(
        self, velocity_s: torch.Tensor, velocity_n: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.acceleration_s[:] = (velocity_s - self.previous_velocity_s) / dt
        self.acceleration_n[:] = (velocity_n - self.previous_velocity_n) / dt
        self.previous_velocity_s[:] = velocity_s
        self.previous_velocity_n[:] = velocity_n
        return self.acceleration_s, self.acceleration_n

    def reset(
        self,
        velocity_s: torch.Tensor,
        velocity_n: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.previous_velocity_s[ids] = velocity_s
        self.previous_velocity_n[ids] = velocity_n
        self.acceleration_s[ids] = 0.0
        self.acceleration_n[ids] = 0.0


def zmp_from_hand_force(
    com_s: torch.Tensor,
    com_n: torch.Tensor,
    com_acceleration_s: torch.Tensor,
    com_acceleration_n: torch.Tensor,
    handle_s: torch.Tensor,
    handle_n: torch.Tensor,
    hand_force_s: torch.Tensor,
    hand_force_n: torch.Tensor,
    robot_mass: torch.Tensor | float,
    *,
    min_ground_reaction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute flat-ground ZMP for the cart-on-robot hand force."""

    mass = torch.as_tensor(robot_mass, device=com_s.device, dtype=com_s.dtype)
    r_s = mass * com_acceleration_s - hand_force_s
    r_n = mass * (com_acceleration_n + GRAVITY) - hand_force_n
    valid = r_n > min_ground_reaction
    denominator = torch.where(valid, r_n, torch.ones_like(r_n))
    hand_moment_about_com = (
        (handle_s - com_s) * hand_force_n - (handle_n - com_n) * hand_force_s
    )
    zmp_s = com_s + (-com_n * r_s - hand_moment_about_com) / denominator
    zmp_s = torch.where(valid, zmp_s, torch.zeros_like(zmp_s))
    return zmp_s, r_s, r_n, valid


def _cross_2d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def convex_support_margin(
    support_points: torch.Tensor,
    query_point: torch.Tensor,
    point_mask: torch.Tensor | None = None,
    *,
    tolerance: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance from points to batched convex support polygons.

    ``support_points`` may be unordered.  All directed point pairs whose other
    valid points lie to their left form candidate hull half-spaces.  The minimum
    inward distance is positive inside and negative outside the convex hull.
    """

    if support_points.ndim != 3 or support_points.shape[-1] != 2:
        raise ValueError("support_points must have shape [N, K, 2]")
    if query_point.shape != (support_points.shape[0], 2):
        raise ValueError("query_point must have shape [N, 2]")
    num_envs, num_points, _ = support_points.shape
    if point_mask is None:
        point_mask = torch.ones(
            (num_envs, num_points), device=support_points.device, dtype=torch.bool
        )
    if point_mask.shape != (num_envs, num_points):
        raise ValueError("point_mask must have shape [N, K]")

    starts = support_points[:, :, None, :]  # [N, i, 1, 2]
    edges = support_points[:, None, :, :] - starts  # [N, i, j, 2]
    lengths = torch.linalg.vector_norm(edges, dim=-1)
    # Compute the batched cross products directly by component.  This avoids
    # materializing [N, i, j, k, 2] vectors while preserving the original
    # unordered-edge convex-hull test.
    edge_x = edges[..., 0, None]
    edge_y = edges[..., 1, None]
    point_x = support_points[:, None, None, :, 0] - support_points[:, :, None, None, 0]
    point_y = support_points[:, None, None, :, 1] - support_points[:, :, None, None, 1]
    side = edge_x * point_y - edge_y * point_x
    other_valid = point_mask[:, None, None, :]
    all_left = torch.all((side >= -tolerance) | ~other_valid, dim=-1)
    endpoints_valid = point_mask[:, :, None] & point_mask[:, None, :]
    candidate = endpoints_valid & (lengths > tolerance) & all_left

    vector_to_query = query_point[:, None, None, :] - starts
    distances = _cross_2d(edges, vector_to_query) / torch.clamp(lengths, min=tolerance)
    infinity = torch.full_like(distances, torch.inf)
    margin = torch.min(torch.where(candidate, distances, infinity), dim=-1).values
    margin = torch.min(margin, dim=-1).values

    # At least three non-collinear valid points are required for a polygon.
    has_three_points = torch.sum(point_mask, dim=-1) >= 3
    area_witness = torch.amax(torch.abs(side), dim=(-1, -2, -3)) > tolerance
    valid = has_three_points & area_witness & torch.isfinite(margin)
    margin = torch.where(valid, margin, torch.zeros_like(margin))
    return margin, valid


def foot_support_polygon(
    foot_position_w: torch.Tensor,
    foot_quaternion_wxyz: torch.Tensor,
    foot_contact: torch.Tensor,
    path_origin_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
    *,
    foot_half_length: float,
    foot_half_width: float,
    foot_center_offset_x: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return actual-pose foot corners, contact mask, and support center."""

    if foot_position_w.ndim != 3 or foot_position_w.shape[1:] != (2, 3):
        raise ValueError("foot_position_w must have shape [N,2,3]")
    if foot_quaternion_wxyz.shape != (foot_position_w.shape[0], 2, 4):
        raise ValueError("foot quaternion must have shape [N,2,4]")
    if foot_contact.shape != (foot_position_w.shape[0], 2):
        raise ValueError("foot_contact must have shape [N,2]")
    if foot_half_length <= 0.0 or foot_half_width <= 0.0:
        raise ValueError("foot half dimensions must be positive")
    if not math.isfinite(foot_center_offset_x):
        raise ValueError("foot center offset must be finite")
    local_corners = getattr(foot_support_polygon, "_local_corners", None)
    cache_key = (foot_half_length, foot_half_width, foot_center_offset_x)
    if (
        local_corners is None
        or getattr(foot_support_polygon, "_local_corners_key", None) != cache_key
        or local_corners.device != foot_position_w.device
        or local_corners.dtype != foot_position_w.dtype
    ):
        local_corners = torch.tensor(
            (
                (foot_center_offset_x + foot_half_length, foot_half_width, 0.0),
                (foot_center_offset_x - foot_half_length, foot_half_width, 0.0),
                (foot_center_offset_x - foot_half_length, -foot_half_width, 0.0),
                (foot_center_offset_x + foot_half_length, -foot_half_width, 0.0),
            ),
            device=foot_position_w.device,
            dtype=foot_position_w.dtype,
        )
        foot_support_polygon._local_corners = local_corners
        foot_support_polygon._local_corners_key = cache_key
        local_center = torch.zeros(
            (1, 1, 3), device=foot_position_w.device, dtype=foot_position_w.dtype
        )
        local_center[..., 0] = foot_center_offset_x
        foot_support_polygon._local_center = local_center
    else:
        local_center = foot_support_polygon._local_center
    local_corners = local_corners.view(1, 1, 4, 3).expand(
        foot_position_w.shape[0], 2, -1, -1
    )
    world_corners = foot_position_w[:, :, None, :] + quat_apply_wxyz(
        foot_quaternion_wxyz[:, :, None, :].expand(-1, -1, 4, -1), local_corners
    )
    relative = world_corners - path_origin_w[:, None, None, :]
    corner_s = torch.sum(relative * path_tangent_w[:, None, None, :], dim=-1)
    corner_y = torch.sum(relative * path_lateral_w[:, None, None, :], dim=-1)
    points = torch.stack((corner_s, corner_y), dim=-1).reshape(-1, 8, 2)
    point_mask = foot_contact[:, :, None].expand(-1, -1, 4).reshape(-1, 8)
    contact_count = torch.sum(foot_contact, dim=-1, keepdim=True)
    foot_center_w = foot_position_w + quat_apply_wxyz(foot_quaternion_wxyz, local_center)
    support_center = torch.sum(
        foot_center_w * foot_contact[..., None].to(foot_position_w.dtype), dim=1
    ) / torch.clamp(contact_count, min=1).to(foot_position_w.dtype)
    support_center = torch.where(
        (contact_count > 0), support_center, torch.zeros_like(support_center)
    )
    return points, point_mask, support_center


__all__ = [
    "AnalyticForceCfg",
    "AnalyticHandleForceState",
    "FAT2Cfg",
    "GRAVITY",
    "RickshawMassProperties",
    "SupportPolygonCfg",
    "ZMPCfg",
    "ZMPKinematicState",
    "analytic_handle_force",
    "articulation_center_of_mass",
    "combine_mass_properties",
    "convex_support_margin",
    "effective_cart_mass",
    "effective_wheel_damping",
    "fat2_reference_angle",
    "force_consistency",
    "foot_support_polygon",
    "parallel_axis_inertia",
    "quat_apply_wxyz",
    "rickshaw_pitch_from_quaternion",
    "rolling_resistance_force",
    "sagittal_com_radius",
    "torso_pitch_from_world_vertical",
    "update_analytic_handle_force_state",
    "zmp_from_hand_force",
]
