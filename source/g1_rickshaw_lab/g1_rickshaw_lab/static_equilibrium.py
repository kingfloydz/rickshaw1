"""Load and validate the certified flat-ground rest pose."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import G1_JOINT_ORDER
from .project_paths import CONFIG_ROOT
from .rickshaw_spec import HITCH_HEIGHT_RANGE

STATIC_REST_POSE_SCHEMA_VERSION = 11
STATIC_REST_POSE_PATH = CONFIG_ROOT / "static_rest_poses.json"
_MODEL_SIGNATURE_DECIMALS = 10


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
        float(record["equality_position_error"]) > 0.003
        or float(record["support_height_error"]) > 0.003
        or not HITCH_HEIGHT_RANGE[0] <= float(record["hitch_height"]) <= HITCH_HEIGHT_RANGE[1]
        or float(record["acceleration_error"]) > 0.5
        or float(record["actuator_torque_ratio"]) > 0.86
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


__all__ = [
    "MujocoStaticEquilibrium",
    "STATIC_REST_POSE_PATH",
    "load_mujoco_static_equilibrium",
    "save_mujoco_static_equilibrium",
]
