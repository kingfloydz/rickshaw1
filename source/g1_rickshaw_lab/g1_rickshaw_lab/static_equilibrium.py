"""Fixed-contact statics shared by reset generation and runtime loading.

The hand-force convention is robot-on-cart. Wheel contact forces are
ground-on-cart. Components use the path frame ``(s, l, n)`` where ``s`` is
forward, ``l`` is left, and ``n`` is up.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets.g1_dex1 import (
    G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS,
    G1_JOINT_EFFORT_LIMITS,
)
from .configuration import G1_JOINT_ORDER
from .project_paths import CONFIG_ROOT
from .rickshaw_spec import (
    HITCH_HALF_WIDTH,
    HITCH_HEIGHT_RANGE,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    WHEEL_RADIUS,
    WHEEL_TRACK,
)

STATIC_REST_POSE_SCHEMA_VERSION = 11
STATIC_REST_POSE_PATH = CONFIG_ROOT / "static_rest_poses.json"
_MODEL_SIGNATURE_DECIMALS = 10


@dataclass(frozen=True)
class FixedContactStaticSolution:
    """Scalar fixed-contact solution for two hitches and two passive wheels."""

    handle_forces_sln: tuple[tuple[float, float, float], tuple[float, float, float]]
    wheel_contact_forces_sln: tuple[tuple[float, ...], tuple[float, ...]]
    cart_force_residual_sln: tuple[float, float, float]
    cart_moment_residual_sln: tuple[float, float, float]


@dataclass(frozen=True)
class MujocoStaticEquilibrium:
    """One MuJoCo equilibrium used directly by the reset event."""

    qpos: np.ndarray
    joint_actuator_torque: np.ndarray
    equality_position_error: float
    support_height_error: float
    hitch_height: float
    acceleration_error: float
    actuator_torque_ratio: float


@dataclass(frozen=True)
class MujocoStaticSolverCfg:
    multistarts: int = 50
    screening_candidates: int = 10
    refinement_candidates: int = 10
    screening_max_nfev: int = 100
    refinement_max_nfev: int = 1000
    max_nfev: int = 5000
    position_scale: float = 1.0e-5
    support_scale: float = 5.0e-5
    foot_tilt_scale: float = 0.01
    contact_penetration: float = -2.0e-4
    constraint_preload: float = 0.0002
    support_friction: float = 1.0
    support_force_tolerance: float = 1.0
    unactuated_force_scale: float = 1.0
    torque_barrier_fraction: float = 0.6
    posture_scale: float = 1.0
    root_orientation_scale: float = 0.5
    position_tolerance: float = 0.003
    support_tolerance: float = 0.003
    hitch_height_range: tuple[float, float] = HITCH_HEIGHT_RANGE
    hitch_height_margin: float = 0.005
    hitch_height_scale: float = 1.0e-3
    acceleration_tolerance: float = 0.5
    actuator_torque_ratio_tolerance: float = 0.86


def _model_signature(model: Any) -> str:
    import mujoco

    digest = hashlib.sha256()
    digest.update(f"{model.nq}:{model.nv}:{model.nu}:{model.nbody}:{model.njnt}:{model.ngeom}:{model.nsite}".encode())
    named_objects = (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        (mujoco.mjtObj.mjOBJ_EQUALITY, model.neq),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    )
    for object_type, count in named_objects:
        names = [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]
        digest.update(json.dumps(names, separators=(",", ":")).encode())
    for value in (
        np.asarray(
            (
                model.opt.timestep,
                model.opt.iterations,
                model.opt.ls_iterations,
                model.opt.ccd_iterations,
            )
        ),
        model.qpos0,
        model.body_mass,
        model.body_inertia,
        model.body_ipos,
        model.body_iquat,
        model.jnt_type,
        model.jnt_pos,
        model.jnt_axis,
        model.jnt_range,
        model.dof_damping,
        model.dof_armature,
        model.geom_type,
        model.geom_size,
        model.geom_pos,
        model.geom_quat,
        model.geom_contype,
        model.geom_conaffinity,
        model.geom_condim,
        model.geom_priority,
        model.geom_friction,
        model.geom_solref,
        model.geom_solimp,
        model.site_pos,
        model.site_quat,
        model.eq_type,
        model.eq_obj1id,
        model.eq_obj2id,
        model.eq_data,
        model.eq_solref,
        model.eq_solimp,
        model.actuator_trnid,
        model.actuator_gainprm,
        model.actuator_biasprm,
        model.actuator_forcerange,
    ):
        source = np.asarray(value)
        if np.issubdtype(source.dtype, np.floating):
            array = np.round(source.astype("<f8"), decimals=_MODEL_SIGNATURE_DECIMALS)
            array[array == 0.0] = 0.0
            kind = b"f"
        elif np.issubdtype(source.dtype, np.integer):
            array = source.astype("<i8")
            kind = b"i"
        else:
            raise TypeError(f"unsupported model signature dtype: {source.dtype}")
        digest.update(kind)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_mujoco_static_equilibrium(
    model: Any,
    solution: MujocoStaticEquilibrium,
    path: str | Path = STATIC_REST_POSE_PATH,
) -> Path:
    """Persist the certified flat-ground rest pose."""

    payload = {
        "schema_version": STATIC_REST_POSE_SCHEMA_VERSION,
        "model_signature": _model_signature(model),
        "solution": {
            "qpos": solution.qpos.tolist(),
            "joint_actuator_torque": solution.joint_actuator_torque.tolist(),
            "equality_position_error": solution.equality_position_error,
            "support_height_error": solution.support_height_error,
            "hitch_height": solution.hitch_height,
            "acceleration_error": solution.acceleration_error,
            "actuator_torque_ratio": solution.actuator_torque_ratio,
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_mujoco_static_equilibrium(
    model: Any,
    path: str | Path = STATIC_REST_POSE_PATH,
) -> MujocoStaticEquilibrium:
    """Load the flat-ground rest pose certified for the compiled model."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATIC_REST_POSE_SCHEMA_VERSION:
        raise ValueError(f"unsupported rest-pose schema in {source}")
    if payload.get("model_signature") != _model_signature(model):
        raise ValueError(f"rest-pose model signature does not match {source}")
    record = payload.get("solution")
    if not isinstance(record, dict):
        raise ValueError(f"rest-pose solution is missing from {source}")

    cfg = MujocoStaticSolverCfg()
    qpos = np.asarray(record["qpos"], dtype=float)
    joint_actuator_torque = np.asarray(record["joint_actuator_torque"], dtype=float)
    scalars = np.asarray(
        [
            record["equality_position_error"],
            record["support_height_error"],
            record["hitch_height"],
            record["acceleration_error"],
            record["actuator_torque_ratio"],
        ],
        dtype=float,
    )
    if (
        qpos.shape != (model.nq,)
        or joint_actuator_torque.shape != (len(G1_JOINT_ORDER),)
        or not np.all(np.isfinite(np.concatenate((qpos, joint_actuator_torque, scalars))))
    ):
        raise ValueError(f"invalid rest-pose record in {source}")
    if (
        float(record["equality_position_error"]) > cfg.position_tolerance
        or float(record["support_height_error"]) > cfg.support_tolerance
        or not cfg.hitch_height_range[0] <= float(record["hitch_height"]) <= cfg.hitch_height_range[1]
        or float(record["acceleration_error"]) > cfg.acceleration_tolerance
        or float(record["actuator_torque_ratio"]) > cfg.actuator_torque_ratio_tolerance
    ):
        raise ValueError(f"rest-pose acceptance metrics failed in {source}")
    return MujocoStaticEquilibrium(
        qpos=qpos,
        joint_actuator_torque=joint_actuator_torque,
        equality_position_error=float(record["equality_position_error"]),
        support_height_error=float(record["support_height_error"]),
        hitch_height=float(record["hitch_height"]),
        acceleration_error=float(record["acceleration_error"]),
        actuator_torque_ratio=float(record["actuator_torque_ratio"]),
    )


def fixed_contact_static_components(
    *,
    gravity_tangent: Any,
    gravity_normal: Any,
    com_s: Any,
    com_l: Any,
    com_n: Any,
    handle_s: Any,
    handle_n: Any,
    hitch_half_width: float,
    wheel_track: float,
) -> tuple[tuple[tuple[Any, ...], tuple[Any, ...]], tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Allocate the closed-chain static forces using passive-wheel mechanics.

    The wheel bearings are passive, so a zero-speed equilibrium cannot rely on
    wheel tangent force: the two hitches carry the cart's tangential load.
    Equal normal hand loading is selected from the redundant lateral load
    family; the wheels carry the lateral-CoM roll moment.  A lateral CoM offset
    gives unequal hand tangent forces to cancel yaw.  The corresponding joint
    torque is affine in ``F_s,left - F_s,right`` and is stored as a separate
    reset-library basis.

    Inputs may be floats, NumPy arrays, or Torch tensors.  The function uses
    only elementwise arithmetic so the exact same equations can be evaluated
    offline and for a batch of randomized runtime environments.
    """

    zero = gravity_tangent * 0.0
    hand_tangent_total = gravity_tangent
    gravity_pitch_moment = com_s * gravity_normal - com_n * gravity_tangent
    hand_normal_total = (handle_n * hand_tangent_total + gravity_pitch_moment) / handle_s

    hand_tangent_difference = com_l * gravity_tangent / hitch_half_width
    hand_normal_difference = zero
    wheel_normal_total = gravity_normal - hand_normal_total
    wheel_normal_difference = 2.0 * com_l * gravity_normal / wheel_track

    left_hand = (
        0.5 * (hand_tangent_total + hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total + hand_normal_difference),
    )
    right_hand = (
        0.5 * (hand_tangent_total - hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total - hand_normal_difference),
    )
    left_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total + wheel_normal_difference),
    )
    right_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total - wheel_normal_difference),
    )
    return (left_hand, right_hand), (left_wheel, right_wheel)


def solve_fixed_contact_statics(
    *,
    mass: float,
    com_from_axle_sln: tuple[float, float, float],
    handle_from_axle_sn: tuple[float, float],
    hitch_half_width: float,
    wheel_track: float,
    gravity: float = 9.81,
) -> FixedContactStaticSolution:
    """Return and independently verify one scalar fixed-contact equilibrium."""

    scalars = (
        mass,
        *com_from_axle_sln,
        *handle_from_axle_sn,
        hitch_half_width,
        wheel_track,
        gravity,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("fixed-contact statics inputs must be finite")
    if mass <= 0.0 or gravity <= 0.0:
        raise ValueError("mass and gravity must be positive")
    if handle_from_axle_sn[0] <= 0.0:
        raise ValueError("handle tangent offset from the axle must be positive")
    if hitch_half_width <= 0.0 or wheel_track <= 0.0:
        raise ValueError("hitch half-width and wheel track must be positive")

    gravity_tangent = 0.0
    gravity_normal = mass * gravity
    hand_forces, wheel_forces = fixed_contact_static_components(
        gravity_tangent=gravity_tangent,
        gravity_normal=gravity_normal,
        com_s=com_from_axle_sln[0],
        com_l=com_from_axle_sln[1],
        com_n=com_from_axle_sln[2],
        handle_s=handle_from_axle_sn[0],
        handle_n=handle_from_axle_sn[1],
        hitch_half_width=hitch_half_width,
        wheel_track=wheel_track,
    )
    hand_forces = tuple(tuple(float(value) for value in row) for row in hand_forces)
    wheel_forces = tuple(tuple(float(value) for value in row) for row in wheel_forces)

    gravity_force = (-gravity_tangent, 0.0, -gravity_normal)
    force_residual = tuple(
        gravity_force[axis] + sum(force[axis] for force in hand_forces) + sum(force[axis] for force in wheel_forces)
        for axis in range(3)
    )
    com_s, com_l, com_n = com_from_axle_sln
    handle_s, handle_n = handle_from_axle_sn
    gravity_moment = (
        -com_l * gravity_normal,
        com_s * gravity_normal - com_n * gravity_tangent,
        com_l * gravity_tangent,
    )
    moment_residual = [float(value) for value in gravity_moment]
    for lateral, force in zip((hitch_half_width, -hitch_half_width), hand_forces, strict=True):
        force_s, force_l, force_n = force
        moment_residual[0] += lateral * force_n - handle_n * force_l
        moment_residual[1] += handle_n * force_s - handle_s * force_n
        moment_residual[2] += handle_s * force_l - lateral * force_s
    for lateral, force in zip((0.5 * wheel_track, -0.5 * wheel_track), wheel_forces, strict=True):
        force_s, _force_l, force_n = force
        moment_residual[0] += lateral * force_n
        moment_residual[2] += -lateral * force_s

    return FixedContactStaticSolution(
        handle_forces_sln=hand_forces,  # type: ignore[arg-type]
        wheel_contact_forces_sln=wheel_forces,  # type: ignore[arg-type]
        cart_force_residual_sln=tuple(float(value) for value in force_residual),
        cart_moment_residual_sln=tuple(moment_residual),
    )


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _rpy_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        (
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


def _joint_qpos_address(model: Any, name: str) -> int:
    return int(model.joint(name).qposadr[0])


def _joint_dof_address(model: Any, name: str) -> int:
    return int(model.joint(name).dofadr[0])


def _nominal_qpos(model: Any) -> np.ndarray:
    """Produce one deterministic IK seed; it is never used as a reset state."""

    import mujoco

    qpos = model.qpos0.copy()
    robot_root = _joint_qpos_address(model, "robot/floating_base_joint")
    rickshaw_root = _joint_qpos_address(model, "rickshaw/floating_base_joint")
    qpos[robot_root : robot_root + 7] = (0.0, 0.0, 0.7837, 1.0, 0.0, 0.0, 0.0)
    arm_seed = {
        "left_shoulder_pitch_joint": 0.364318342451,
        "right_shoulder_pitch_joint": 0.364309963414,
        "left_shoulder_roll_joint": -0.016424064972,
        "right_shoulder_roll_joint": 0.016404723642,
        "left_shoulder_yaw_joint": 0.836675882550,
        "right_shoulder_yaw_joint": -0.836698499844,
        "left_elbow_joint": 0.366720603917,
        "right_elbow_joint": 0.366732610982,
        "left_wrist_roll_joint": -1.201784461031,
        "right_wrist_roll_joint": 1.201791820224,
        "left_wrist_pitch_joint": 0.702616905883,
        "right_wrist_pitch_joint": 0.702604600279,
        "left_wrist_yaw_joint": -1.376590436886,
        "right_wrist_yaw_joint": 1.376571598415,
    }
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith("robot/") or name.endswith("floating_base_joint"):
            continue
        short_name = name.removeprefix("robot/")
        qpos[int(model.jnt_qposadr[joint_id])] = G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS.get(
            short_name, arm_seed.get(short_name, 0.0)
        )

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    grasp_midpoint = 0.5 * (data.site("robot/left_grasp_site").xpos + data.site("robot/right_grasp_site").xpos)
    hitch_midpoint = np.array((1.664929, 0.0, 0.105747))
    wheel_radius = 0.3
    low, high = 0.0, 0.7
    for _ in range(48):
        angle = 0.5 * (low + high)
        height = (
            wheel_radius * (1.0 - math.cos(angle))
            + math.sin(angle) * hitch_midpoint[0]
            + math.cos(angle) * hitch_midpoint[2]
        )
        if height < grasp_midpoint[2]:
            low = angle
        else:
            high = angle
    angle = 0.5 * (low + high)
    quat = _quat_from_rpy(np.array((0.0, -angle, 0.0)))
    rotation = np.array(
        (
            (math.cos(angle), 0.0, -math.sin(angle)),
            (0.0, 1.0, 0.0),
            (math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    cart_position = grasp_midpoint - rotation @ hitch_midpoint
    qpos[rickshaw_root : rickshaw_root + 7] = (*tuple(cart_position), *tuple(quat))
    return qpos


def _pose_multistart_seeds(
    base: np.ndarray,
    nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_start: int,
    count: int,
) -> tuple[np.ndarray, ...]:
    """Build the deterministic mirrored seed family used by the static solve."""

    if count < 2:
        raise ValueError("static equilibrium requires at least two multistarts")
    root_pitch_offsets = (0.0, -0.04, 0.04, -0.08, 0.08)
    root_height_offsets = (0.0, -0.025, -0.025, 0.025, 0.025)
    arm_scale = np.asarray((0.25, 0.20, 0.30, 0.25, 0.30, 0.30, 0.30))
    right_symmetry = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
    seeds: list[np.ndarray] = []
    alternate = nominal.copy()
    if np.allclose(base, nominal):
        alternate[2] -= 0.05
        alternate[4] -= 0.10
        alternate[joint_start] += 0.15
        alternate[joint_start + 3] += 0.10
        alternate[joint_start + 7] += 0.15
        alternate[joint_start + 10] += 0.10
    for index in range(count):
        seed = (base if index % 2 == 0 else alternate).copy()
        profile = index // 2
        offset_index = profile % len(root_pitch_offsets)
        scale = 1.0 + profile // len(root_pitch_offsets)
        seed[4] += scale * root_pitch_offsets[offset_index]
        seed[2] += scale * root_height_offsets[offset_index]
        if profile:
            rng = np.random.default_rng(26_212 + profile)
            left_delta = (-1.0 if profile % 2 == 0 else 1.0) * arm_scale * rng.standard_normal(7)
            seed[joint_start : joint_start + 7] += left_delta
            seed[joint_start + 7 : joint_start + 14] += right_symmetry * left_delta
        seeds.append(np.clip(seed, np.nextafter(lower, upper), np.nextafter(upper, lower)))
    return tuple(seeds)


def solve_mujoco_static_equilibrium(
    model: Any,
    *,
    cfg: MujocoStaticSolverCfg | None = None,
) -> MujocoStaticEquilibrium:
    """Solve a fixed-contact equilibrium with MuJoCo dynamics.

    The optimization has no dynamic settling phase. Inverse dynamics solves the
    pose, explicit contact reactions, and actuator torque together.
    Forward dynamics validates the pose with direct joint generalized forces;
    the position actuators remain neutral during acceptance.
    """

    import mujoco
    from scipy.optimize import least_squares

    if cfg is None:
        cfg = MujocoStaticSolverCfg()
    q_seed = _nominal_qpos(model)

    robot_root_q = _joint_qpos_address(model, "robot/floating_base_joint")
    cart_root_q = _joint_qpos_address(model, "rickshaw/floating_base_joint")
    robot_root_v = _joint_dof_address(model, "robot/floating_base_joint")
    cart_root_v = _joint_dof_address(model, "rickshaw/floating_base_joint")
    robot_joint_ids = [
        index
        for index in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").startswith("robot/")
        and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    ]
    robot_joint_q = np.array([model.jnt_qposadr[index] for index in robot_joint_ids], dtype=int)
    robot_joint_v = np.array([model.jnt_dofadr[index] for index in robot_joint_ids], dtype=int)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "" for index in robot_joint_ids]
    if tuple(name.removeprefix("robot/") for name in joint_names) != G1_JOINT_ORDER:
        raise ValueError("MuJoCo robot joint order does not match the 29-joint policy contract")
    arm_offsets = np.asarray(
        [
            index
            for index, name in enumerate(G1_JOINT_ORDER)
            if name not in G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS
        ],
        dtype=int,
    )
    if arm_offsets.size != 14:
        raise ValueError("G1 static solve requires exactly 14 arm joints")
    arm_joint_ids = np.asarray(robot_joint_ids, dtype=int)[arm_offsets]
    arm_joint_q = robot_joint_q[arm_offsets]
    effort_limits = np.asarray(G1_JOINT_EFFORT_LIMITS)
    robot_actuator_ids = np.array(
        [
            actuator_id
            for joint_id in robot_joint_ids
            for actuator_id in range(model.nu)
            if model.actuator_trnid[actuator_id, 0] == joint_id
        ],
        dtype=int,
    )
    if robot_actuator_ids.size not in (0, len(robot_joint_ids)):
        raise ValueError("static model must have either zero or one actuator per robot joint")
    wheel_v = np.array(
        [
            _joint_dof_address(model, "rickshaw/left_wheel_joint"),
            _joint_dof_address(model, "rickshaw/right_wheel_joint"),
        ]
    )
    unactuated_v = np.concatenate(
        (
            np.arange(robot_root_v, robot_root_v + 6),
            np.arange(cart_root_v, cart_root_v + 6),
            wheel_v,
        )
    )
    foot_geoms: list[int] = []
    for body_name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link"):
        body_id = model.body(body_name).id
        first = model.body_geomadr[body_id]
        count = model.body_geomnum[body_id]
        foot_geoms.extend(
            index for index in range(first, first + count) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
    wheel_geoms = [
        int(model.body_geomadr[model.body(name).id])
        for name in ("rickshaw/left_wheel_link", "rickshaw/right_wheel_link")
    ]
    if len(foot_geoms) != 8:
        raise ValueError(f"expected eight foot contact spheres, got {len(foot_geoms)}")

    static_cart = solve_fixed_contact_statics(
        mass=RICKSHAW_TOTAL_MASS,
        com_from_axle_sln=(
            RICKSHAW_CENTER_OF_MASS[0],
            RICKSHAW_CENTER_OF_MASS[1],
            RICKSHAW_CENTER_OF_MASS[2] - WHEEL_RADIUS,
        ),
        handle_from_axle_sn=(HITCH_X, HITCH_Z - WHEEL_RADIUS),
        hitch_half_width=HITCH_HALF_WIDTH,
        wheel_track=WHEEL_TRACK,
    )
    def pose_from_qpos(qpos: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                qpos[robot_root_q : robot_root_q + 3],
                _rpy_from_quat(qpos[robot_root_q + 3 : robot_root_q + 7]),
                qpos[arm_joint_q],
                qpos[cart_root_q : cart_root_q + 3],
                _rpy_from_quat(qpos[cart_root_q + 3 : cart_root_q + 7]),
            )
        )

    pose_seed = pose_from_qpos(q_seed)
    lower = np.full_like(pose_seed, -np.inf)
    upper = np.full_like(pose_seed, np.inf)
    lower[:3], upper[:3] = (-0.2, -0.1, 0.55), (0.2, 0.1, 0.9)
    lower[3:6], upper[3:6] = (-0.35, -0.6, -0.2), (0.35, 0.6, 0.2)
    joint_start = 6
    joint_end = joint_start + len(arm_joint_ids)
    for offset, joint_id in enumerate(arm_joint_ids):
        if model.jnt_limited[joint_id]:
            lower[joint_start + offset], upper[joint_start + offset] = model.jnt_range[joint_id]
    lower[joint_end : joint_end + 3] = (-2.0, -0.4, -0.05)
    upper[joint_end : joint_end + 3] = (0.5, 0.4, 0.2)
    lower[-3:], upper[-3:] = (-0.35, -0.6, -0.2), (0.35, 0.6, 0.2)
    nominal_pose_seed = pose_from_qpos(_nominal_qpos(model))
    initial_seeds = _pose_multistart_seeds(
        pose_seed,
        nominal_pose_seed,
        lower,
        upper,
        joint_start,
        cfg.multistarts,
    )

    data = mujoco.MjData(model)
    gravity_original = model.opt.gravity.copy()
    geom_contype_original = model.geom_contype.copy()
    geom_conaffinity_original = model.geom_conaffinity.copy()
    eq_active_original = data.eq_active.copy()
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data.eq_active[:] = 0

    def unpack(x: np.ndarray) -> None:
        data.qpos[:] = q_seed
        data.qpos[robot_root_q : robot_root_q + 3] = x[:3]
        data.qpos[robot_root_q + 3 : robot_root_q + 7] = _quat_from_rpy(x[3:6])
        data.qpos[arm_joint_q] = x[joint_start:joint_end]
        data.qpos[cart_root_q : cart_root_q + 3] = x[joint_end : joint_end + 3]
        data.qpos[cart_root_q + 3 : cart_root_q + 7] = _quat_from_rpy(x[-3:])
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.qfrc_applied[:] = 0.0
        if robot_actuator_ids.size:
            data.ctrl[robot_actuator_ids] = data.qpos[robot_joint_q]

    def kinematic_residuals() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        position_errors: list[np.ndarray] = []
        for side in ("left", "right"):
            grasp = data.site(f"robot/{side}_grasp_site")
            hitch = data.site(f"rickshaw/{side}_hitch_site")
            position_errors.append(np.asarray(grasp.xpos) - np.asarray(hitch.xpos))
        # The four spheres on one rigid foot are coplanar contact samples.
        # Constrain the support plane (height, roll, pitch), not four
        # independent point heights.
        foot_height = np.array(
            [
                np.mean(
                    [data.geom_xpos[index, 2] - model.geom_size[index, 0] for index in foot_geoms[offset : offset + 4]]
                )
                for offset in (0, 4)
            ]
        )
        foot_tilt = np.concatenate(
            [
                data.body(body_name).xmat.reshape(3, 3)[:2, 2]
                for body_name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
            ]
        )
        wheel_height = np.array([data.geom_xpos[index, 2] - model.geom_size[index, 0] for index in wheel_geoms])
        handle_center = 0.5 * (data.site("robot/left_grasp_site").xpos + data.site("robot/right_grasp_site").xpos)
        hitch_height = float(handle_center[2])
        torso = data.body("robot/torso_link").xmat.reshape(3, 3)
        torso_pitch = math.atan2(-torso[2, 0], math.hypot(torso[0, 0], torso[1, 0]))
        return (
            np.concatenate(position_errors),
            np.concatenate((foot_height, wheel_height)) - cfg.contact_penetration,
            foot_tilt,
            torso_pitch,
            hitch_height,
        )

    def hitch_height_violation(hitch_height: float) -> float:
        low, high = cfg.hitch_height_range
        return hitch_height - float(
            np.clip(hitch_height, low + cfg.hitch_height_margin, high - cfg.hitch_height_margin)
        )

    def point_force_jacobian(body_id: int, point: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, point, body_id)
        return jacobian.T

    def static_external_forces() -> tuple[np.ndarray, np.ndarray]:
        external = np.zeros(model.nv)
        for side, cart_force in zip(("left", "right"), static_cart.handle_forces_sln, strict=True):
            force = np.asarray(cart_force)
            hitch = data.site(f"rickshaw/{side}_hitch_site")
            grasp = data.site(f"robot/{side}_grasp_site")
            external += point_force_jacobian(
                int(model.site_bodyid[hitch.id]), np.asarray(hitch.xpos)
            ) @ force
            external += point_force_jacobian(
                int(model.site_bodyid[grasp.id]), np.asarray(grasp.xpos)
            ) @ -force

        for body_name, wheel_force in zip(
            ("rickshaw/left_wheel_link", "rickshaw/right_wheel_link"),
            static_cart.wheel_contact_forces_sln,
            strict=True,
        ):
            body = data.body(body_name)
            external += point_force_jacobian(body.id, np.asarray(body.xpos)) @ np.asarray(wheel_force)

        foot_force_map = np.concatenate(
            [
                point_force_jacobian(
                    int(model.geom_bodyid[geom_id]),
                    data.geom_xpos[geom_id] - np.array((0.0, 0.0, model.geom_size[geom_id, 0])),
                )
                for geom_id in foot_geoms
            ],
            axis=1,
        )
        root_target = data.qfrc_inverse[robot_root_v : robot_root_v + 6] - external[robot_root_v : robot_root_v + 6]
        foot_forces = np.linalg.lstsq(foot_force_map[robot_root_v : robot_root_v + 6], root_target, rcond=None)[0]
        external += foot_force_map @ foot_forces
        return external, foot_forces.reshape(len(foot_geoms), 3)

    def residual(x: np.ndarray) -> np.ndarray:
        unpack(x)
        mujoco.mj_forward(model, data)
        position_error, support_error, foot_tilt, torso_pitch, hitch_height = kinematic_residuals()

        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        external_force, foot_forces = static_external_forces()
        # With constraints disabled, MuJoCo returns C(q) for qacc=0.  The
        # actuator force required by M(q)qacc+C(q)=tau+J^T f is therefore
        # qfrc_inverse - external_force.
        required_force = data.qfrc_inverse - external_force
        unactuated_force = required_force[unactuated_v]
        joint_torque = required_force[robot_joint_v]
        torque_ratio = np.abs(joint_torque) / effort_limits
        torque_barrier = np.logaddexp(0.0, 20.0 * (torque_ratio - cfg.torque_barrier_fraction)) / 2.0
        return np.concatenate(
            (
                position_error / cfg.position_scale,
                support_error / cfg.support_scale,
                foot_tilt / cfg.foot_tilt_scale,
                np.asarray((hitch_height_violation(hitch_height) / cfg.hitch_height_scale,)),
                unactuated_force / cfg.unactuated_force_scale,
                torque_barrier,
                np.minimum(foot_forces[:, 2], 0.0) / cfg.unactuated_force_scale,
                np.maximum(
                    np.linalg.norm(foot_forces[:, :2], axis=1) - cfg.support_friction * foot_forces[:, 2],
                    0.0,
                )
                / cfg.unactuated_force_scale,
                (x[joint_start:joint_end] - q_seed[arm_joint_q]) / cfg.posture_scale,
                np.array(
                    (
                        torso_pitch / cfg.root_orientation_scale,
                        x[3] / cfg.root_orientation_scale,
                        x[4] / cfg.root_orientation_scale,
                        x[0] / 0.05,
                        x[1] / 0.03,
                    )
                ),
            )
        )

    try:
        candidates: list[tuple[tuple[float, ...], Any, np.ndarray]] = []
        ranked_seeds = sorted(initial_seeds, key=lambda seed: float(np.linalg.norm(residual(seed))))
        for seed in ranked_seeds[: cfg.screening_candidates]:
            candidate = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                max_nfev=cfg.screening_max_nfev,
                xtol=1.0e-9,
                ftol=1.0e-9,
                gtol=1.0e-9,
                x_scale="jac",
            )
            unpack(candidate.x)
            mujoco.mj_forward(model, data)
            position_error, support_error, foot_tilt, _, hitch_height = kinematic_residuals()
            data.qacc[:] = 0.0
            mujoco.mj_inverse(model, data)
            candidate_external, candidate_foot_forces = static_external_forces()
            candidate_force = data.qfrc_inverse - candidate_external
            constraint_error = max(
                float(np.linalg.norm(position_error, ord=np.inf) / cfg.position_scale),
                float(np.linalg.norm(support_error, ord=np.inf) / cfg.support_scale),
                float(np.linalg.norm(foot_tilt, ord=np.inf) / cfg.foot_tilt_scale),
                abs(hitch_height_violation(hitch_height)) / cfg.hitch_height_scale,
                float(np.linalg.norm(candidate_force[unactuated_v], ord=np.inf) / cfg.unactuated_force_scale),
                float(max(0.0, -np.min(candidate_foot_forces[:, 2])) / cfg.support_force_tolerance),
            )
            torque_ratio = float(np.max(np.abs(candidate_force[robot_joint_v]) / effort_limits))
            candidates.append(
                (
                    (
                        float(constraint_error > 1.0),
                        constraint_error,
                        float(torque_ratio > cfg.actuator_torque_ratio_tolerance),
                        torque_ratio,
                        float(np.linalg.norm(candidate.fun)),
                    ),
                    candidate,
                    candidate_foot_forces.copy(),
                )
            )

        # MuJoCo contacts and connections are soft constraints. Refine the
        # rigid-load solution against their actual inverse dynamics so the
        # runtime model starts at qacc=0 without settling or injected forces.
        model.geom_contype[:] = geom_contype_original
        model.geom_conaffinity[:] = geom_conaffinity_original
        data = mujoco.MjData(model)
        data.eq_active[:] = eq_active_original
        constraint_lower = lower.copy()
        constraint_lower[2] -= 2.0 * cfg.constraint_preload

        def constraint_inverse(x: np.ndarray) -> np.ndarray:
            unpack(x)
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            mujoco.mj_inverse(model, data)
            return data.qfrc_inverse

        def evaluate_candidate(
            result: Any, foot_forces: np.ndarray
        ) -> tuple[tuple[float, ...], MujocoStaticEquilibrium, dict[str, Any]]:
            constraint_seed = result.x.copy()
            constraint_seed[2] -= cfg.constraint_preload
            constraint_seed[joint_end + 2] -= cfg.constraint_preload
            constraint_seed = np.clip(constraint_seed, constraint_lower, upper)
            refinement_diagnostics: list[tuple[float, float]] = []

            def record(force: np.ndarray) -> None:
                refinement_diagnostics.append(
                    (
                        float(np.linalg.norm(force[unactuated_v], ord=np.inf)),
                        float(np.max(np.abs(force[robot_joint_v]) / effort_limits)),
                    )
                )

            def refine(
                seed: np.ndarray,
                *,
                force_scale: float,
                posture_scale: float,
                difference_step: float,
            ) -> np.ndarray:
                def refine_residual(x: np.ndarray) -> np.ndarray:
                    inverse_force = constraint_inverse(x)
                    position_error, support_error, foot_tilt, _, hitch_height = kinematic_residuals()
                    terms = [
                        inverse_force[unactuated_v] / force_scale,
                        inverse_force[robot_joint_v] / effort_limits,
                        position_error / cfg.position_scale,
                        support_error / cfg.support_scale,
                        foot_tilt / cfg.foot_tilt_scale,
                        np.asarray((hitch_height_violation(hitch_height) / cfg.hitch_height_scale,)),
                        (x[joint_start:joint_end] - constraint_seed[joint_start:joint_end])
                        / posture_scale,
                    ]
                    excess = np.maximum(
                        np.abs(inverse_force[robot_joint_v])
                        - cfg.torque_barrier_fraction * effort_limits,
                        0.0,
                    )
                    terms.insert(2, excess / 0.5)
                    return np.concatenate(terms)

                def absolute_jacobian(x: np.ndarray) -> np.ndarray:
                    value = refine_residual(x)
                    jacobian = np.empty((value.size, x.size))
                    for index in range(x.size):
                        shifted = x.copy()
                        shifted[index] = min(x[index] + difference_step, upper[index])
                        if shifted[index] == x[index]:
                            shifted[index] = max(x[index] - difference_step, constraint_lower[index])
                        jacobian[:, index] = (refine_residual(shifted) - value) / (shifted[index] - x[index])
                    return jacobian

                refined = least_squares(
                    refine_residual,
                    seed,
                    jac=absolute_jacobian,
                    bounds=(constraint_lower, upper),
                    max_nfev=cfg.refinement_max_nfev,
                    xtol=1.0e-13,
                    ftol=1.0e-13,
                    gtol=1.0e-13,
                )
                record(constraint_inverse(refined.x))
                return refined.x

            record(constraint_inverse(constraint_seed))
            refined_pose = constraint_seed
            for force_scale, posture_scale, difference_step in (
                (10.0, 100.0, 1.0e-5),
                (2.0, 200.0, 2.0e-6),
                (0.2, 500.0, 5.0e-7),
                (0.02, 1000.0, 1.0e-7),
                (0.002, 2000.0, 5.0e-8),
                (0.02, 1000.0, 5.0e-8),
            ):
                refined_pose = refine(
                    refined_pose,
                    force_scale=force_scale,
                    posture_scale=posture_scale,
                    difference_step=difference_step,
                )

            inverse_force = constraint_inverse(refined_pose).copy()
            unactuated_error = float(np.linalg.norm(inverse_force[unactuated_v], ord=np.inf))
            inverse_joint_torque = inverse_force[robot_joint_v].copy()

            def forward_acceleration(joint_torque: np.ndarray) -> np.ndarray:
                unpack(refined_pose)
                data.qfrc_applied[robot_joint_v] = joint_torque
                mujoco.mj_forward(model, data)
                return data.qacc.copy()

            torque_bounds = cfg.actuator_torque_ratio_tolerance * effort_limits
            torque_result = least_squares(
                forward_acceleration,
                np.clip(inverse_joint_torque, -torque_bounds, torque_bounds),
                bounds=(-torque_bounds, torque_bounds),
                diff_step=0.01,
                max_nfev=cfg.max_nfev,
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            joint_torque = torque_result.x
            unpack(refined_pose)
            data.qfrc_applied[robot_joint_v] = joint_torque
            mujoco.mj_forward(model, data)
            final_qpos = data.qpos.copy()
            _, _, _, _, final_hitch_height = kinematic_residuals()
            actuator_torque_ratio = float(np.max(np.abs(joint_torque) / effort_limits))
            position_error = max(
                np.linalg.norm(
                    data.site(f"robot/{side}_grasp_site").xpos
                    - data.site(f"rickshaw/{side}_hitch_site").xpos
                )
                for side in ("left", "right")
            )
            support_error = max(
                abs(data.geom_xpos[index, 2] - model.geom_size[index, 0] - cfg.contact_penetration)
                for index in (*foot_geoms, *wheel_geoms)
            )
            acceleration_error = float(np.linalg.norm(data.qacc, ord=np.inf))
            foot_normal_error = float(max(0.0, -np.min(foot_forces[:, 2])))
            foot_friction_error = float(
                max(
                    0.0,
                    np.max(
                        np.linalg.norm(foot_forces[:, :2], axis=1)
                        - cfg.support_friction * foot_forces[:, 2]
                    ),
                )
            )
            wheel_normal_error = float(
                max(0.0, -min(force[2] for force in static_cart.wheel_contact_forces_sln))
            )
            contact_force_error = max(foot_normal_error, foot_friction_error, wheel_normal_error)
            low_height, high_height = cfg.hitch_height_range
            normalized_errors = (
                position_error / cfg.position_tolerance,
                support_error / cfg.support_tolerance,
                0.0 if low_height <= final_hitch_height <= high_height else 2.0,
                acceleration_error / cfg.acceleration_tolerance,
                actuator_torque_ratio / cfg.actuator_torque_ratio_tolerance,
                contact_force_error / cfg.support_force_tolerance,
            )
            score = (
                float(sum(value > 1.0 for value in normalized_errors)),
                float(max(normalized_errors)),
                acceleration_error,
                actuator_torque_ratio,
            )
            equilibrium = MujocoStaticEquilibrium(
                qpos=final_qpos,
                joint_actuator_torque=joint_torque,
                equality_position_error=float(position_error),
                support_height_error=float(support_error),
                hitch_height=final_hitch_height,
                acceleration_error=acceleration_error,
                actuator_torque_ratio=actuator_torque_ratio,
            )
            diagnostics = {
                "inverse": result.message,
                "unactuated": unactuated_error,
                "contact_force": contact_force_error,
                "foot_normal": foot_normal_error,
                "foot_friction": foot_friction_error,
                "wheel_normal": wheel_normal_error,
                "torque_joint": joint_names[int(np.argmax(np.abs(joint_torque) / effort_limits))],
                "refinement": refinement_diagnostics,
                "unactuated_force": inverse_force[unactuated_v].tolist(),
                "lower_margin": float(np.min(refined_pose - constraint_lower)),
                "upper_margin": float(np.min(upper - refined_pose)),
            }
            return score, equilibrium, diagnostics

        candidates.sort(key=lambda item: item[0])
        evaluated = [
            evaluate_candidate(result, forces)
            for _, result, forces in candidates[: cfg.refinement_candidates]
        ]
        valid = [item for item in evaluated if item[0][0] == 0.0]
        if valid:
            score, equilibrium, diagnostics = min(
                valid,
                key=lambda item: (
                    item[1].actuator_torque_ratio,
                    item[0][1],
                    item[0][2],
                ),
            )
        else:
            score, equilibrium, diagnostics = min(evaluated, key=lambda item: item[0])
        if score[0] != 0.0:
            raise RuntimeError(
                "MuJoCo static solve failed: "
                f"inverse={diagnostics['inverse']}; "
                f"position={equilibrium.equality_position_error:.6g}, "
                f"support={equilibrium.support_height_error:.6g}, "
                f"hitch_height={equilibrium.hitch_height:.6g}, "
                f"qacc={equilibrium.acceleration_error:.6g}, "
                f"unactuated={diagnostics['unactuated']:.6g}, "
                f"contact_force={diagnostics['contact_force']:.6g} "
                f"(normal={diagnostics['foot_normal']:.6g}, "
                f"friction={diagnostics['foot_friction']:.6g}, "
                f"wheel={diagnostics['wheel_normal']:.6g}), "
                f"torque_ratio={equilibrium.actuator_torque_ratio:.6g}, "
                f"torque_joint={diagnostics['torque_joint']}, "
                f"refinement={diagnostics['refinement']}, "
                f"unactuated_force={diagnostics['unactuated_force']}, "
                f"lower_margin={diagnostics['lower_margin']:.6g}, "
                f"upper_margin={diagnostics['upper_margin']:.6g}"
            )
        return equilibrium
    finally:
        model.opt.gravity[:] = gravity_original
        model.geom_contype[:] = geom_contype_original
        model.geom_conaffinity[:] = geom_conaffinity_original
        data.eq_active[:] = eq_active_original


__all__ = [
    "FixedContactStaticSolution",
    "MujocoStaticEquilibrium",
    "MujocoStaticSolverCfg",
    "STATIC_REST_POSE_PATH",
    "fixed_contact_static_components",
    "load_mujoco_static_equilibrium",
    "save_mujoco_static_equilibrium",
    "solve_mujoco_static_equilibrium",
    "solve_fixed_contact_statics",
]
