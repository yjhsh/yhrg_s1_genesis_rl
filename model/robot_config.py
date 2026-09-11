"""
Unified configuration module for the YHRG S1 robot.

Centralizes all robot-related constants as the single source of truth for the
project, eliminating duplicated definitions across rl_push_env.py,
model/factories.py and config/scene_config.py.

Robot structure:
  - 6 revolute joints (1joint~6joint) form the manipulator arm
  - 2 prismatic joints (7joint, 8joint) form the gripper
  - base_link is fixed in the world frame

Configuration profiles:
  - INTERACTIVE: for interactive simulation, high-gain + low-torque for precise
    position control
  - RL_PUSH: for RL pushing tasks, low-gain + high-torque for compliant control
    (now unified with INTERACTIVE parameters)
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# ── Project root directory and URDF path ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URDF_PATH: str = os.path.join(
    _PROJECT_ROOT, "asset", "urdf", "yhrg_s1_description", "urdf", "yhrg_s1_v3.urdf"
)

# ── Joint structure ──
JOINT_NAMES: Tuple[str, ...] = (
    "1joint", "2joint", "3joint", "4joint", "5joint", "6joint", "7joint", "8joint",
)
ARM_DOF: int = 6          # joints 1-6
GRIPPER_DOF: int = 2      # joints 7-8
TOTAL_DOF: int = 8
BASE_LINK_NAME: str = "base_link"
EE_LINK_NAME: str = "6_Link"
GRIPPER_LEFT_LINK: str = "7_Link"
GRIPPER_RIGHT_LINK: str = "8_Link"

# ── Joint limits ──
JOINT_LIMITS_LOWER: np.ndarray = np.array([
    -2.967, 0.0, 0.0, -1.57, -1.57, -1.57, 0.0, 0.0,
])
JOINT_LIMITS_UPPER: np.ndarray = np.array([
    2.967, 3.14, 3.14, 1.518, 1.57, 1.57, 0.05, 0.05,
])

# ── PD control gains (RL training: low-gain compliant control) ──
# 6 arm joints: Kp=[10,20,20,10,5,5], Kd=[2,2,2,1,1,0.5]
# 2 gripper joints: kept as low-gain position servos
CONTROL_KP: np.ndarray = np.array([
    10.0, 20.0, 20.0, 10.0, 5.0, 5.0, 400.0, 400.0,
])
CONTROL_KV: np.ndarray = np.array([
    2.0, 2.0, 2.0, 1.0, 1.0, 0.5, 40.0, 40.0,
])

# ── Torque limits (RL training: matched to low-gain PD) ──
# 6 arm joints: [24,24,24,9,9,5] Nm
# 2 gripper joints: kept at 20 Nm
FORCE_LIMITS_LOWER: np.ndarray = np.array([
    -24.0, -24.0, -24.0, -9.0, -9.0, -5.0, -20.0, -20.0,
])
FORCE_LIMITS_UPPER: np.ndarray = np.array([
    24.0, 24.0, 24.0, 9.0, 9.0, 5.0, 20.0, 20.0,
])

# ── Default joint positions ──
NEUTRAL_POSITION: np.ndarray = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])
HOME_POSITION: np.ndarray = np.array([
    0.0, 1.57, 1.57, 0.0, 0.0, 0.0, 0.025, 0.025,
])

# ── RL-push task dedicated default position ──
# j1=-0.053, j2=2.079, j3=0.974, j4=0.429 → EE at (0.294, -0.015, 0.093)
RL_PUSH_DEFAULT_POSITION: np.ndarray = np.array([
    0.0, 1.66, 1.34, 1.0, 0.0, 0.0, 0.0, 0.0,
])


@dataclass
class RobotProfile:
    """机器人控制参数Profile，包含PD增益、力矩限制和默认位置。"""
    kp: np.ndarray
    kv: np.ndarray
    force_lower: np.ndarray
    force_upper: np.ndarray
    default_position: np.ndarray
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kp": self.kp,
            "kv": self.kv,
            "force_lower": self.force_lower,
            "force_upper": self.force_upper,
            "default_position": self.default_position,
            "name": self.name,
        }


def get_interactive_profile() -> RobotProfile:
    """获取交互式仿真控制参数Profile（高增益+低力矩，精确位置控制）。"""
    return RobotProfile(
        name="interactive",
        kp=CONTROL_KP.copy(),
        kv=CONTROL_KV.copy(),
        force_lower=FORCE_LIMITS_LOWER.copy(),
        force_upper=FORCE_LIMITS_UPPER.copy(),
        default_position=NEUTRAL_POSITION.copy(),
    )


def get_rl_push_profile() -> RobotProfile:
    """获取RL推动任务控制参数Profile（统一标准参数 + RL专用默认位置）。"""
    return RobotProfile(
        name="rl_push",
        kp=CONTROL_KP.copy(),
        kv=CONTROL_KV.copy(),
        force_lower=FORCE_LIMITS_LOWER.copy(),
        force_upper=FORCE_LIMITS_UPPER.copy(),
        default_position=RL_PUSH_DEFAULT_POSITION.copy(),
    )


# ── Scene layout parameters ──

# Tabletop
TABLE_POS: Tuple[float, float, float] = (0.0, 0.0, 0.2)      # table center position
TABLE_DIMS: Tuple[float, float, float] = (0.4, 0.5, 0.4)      # table XYZ dimensions
TABLE_TOP_Z: float = TABLE_POS[2] + TABLE_DIMS[2] / 2         # table top Z = 0.4

# Manipulator base frame
ROBOT_BASE_POS: Tuple[float, float, float] = (-0.4, 0.0, 0.4)  # on table top
ROBOT_BASE_QUAT: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Genesis [w,x,y,z], 180° about Z

# Object randomization range
OBJ_XY_LOW: Tuple[float, float] = (-0.16, -0.20)
OBJ_XY_HIGH: Tuple[float, float] = (0.16, 0.20)
OBJ_YAW_RANGE: float = math.pi / 2  # [-90°, 90°]

# Goal randomization range
GOAL_XY_LOW: Tuple[float, float] = (-0.16, -0.20)
GOAL_XY_HIGH: Tuple[float, float] = (0.16, 0.20)

# Object Z-height computation parameters
CUBE_SIZE: Tuple[float, float, float] = (0.04, 0.04, 0.04)
Z_EPS: float = 0.001  # anti-penetration offset

# Out-of-bounds detection range
OBJ_BOUNDS_LOW: Tuple[float, float] = (-0.25, -0.30)
OBJ_BOUNDS_HIGH: Tuple[float, float] = (0.25, 0.30)

# Goal marker (visualization only)
GOAL_MARKER_SIZE: float = 0.04            # side length of the square goal marker (m)
GOAL_MARKER_THICKNESS: float = 0.0001     # thickness of the goal marker (m)
GOAL_MARKER_COLOR: Tuple[float, float, float] = (0.9, 0.1, 0.1)  # red (R, G, B)


@dataclass
class SceneProfile:
    """Scene layout profile, containing table, robot pose and object/goal
    randomization ranges.

    Reserved for future object-randomization extensions: object size, mass, and
    friction parameters can be added as natural fields of SceneProfile.
    """
    table_pos: Tuple[float, float, float]
    table_dims: Tuple[float, float, float]
    table_top_z: float
    robot_base_pos: Tuple[float, float, float]
    robot_base_quat: Tuple[float, float, float, float]
    obj_xy_low: Tuple[float, float]
    obj_xy_high: Tuple[float, float]
    obj_yaw_range: float
    goal_xy_low: Tuple[float, float]
    goal_xy_high: Tuple[float, float]
    obj_z: float  # precomputed object center Z
    obj_bounds_low: Tuple[float, float]
    obj_bounds_high: Tuple[float, float]
    cube_size: Tuple[float, float, float]
    z_eps: float
    # Goal marker (visualization only)
    goal_marker_size: float = GOAL_MARKER_SIZE
    goal_marker_thickness: float = GOAL_MARKER_THICKNESS
    goal_marker_z: float = 0.0            # marker center Z (table top + eps)
    goal_marker_color: Tuple[float, float, float] = GOAL_MARKER_COLOR
    name: str = ""


def get_rl_push_scene_profile() -> SceneProfile:
    """Get the scene-layout profile for the RL pushing task."""
    # Object Z height: table top - object convex-hull min Z + z_eps
    # For a cube centered at origin, min(hull_z) = -cube_size[2]/2
    min_hull_z = -CUBE_SIZE[2] / 2
    obj_z = TABLE_TOP_Z - min_hull_z + Z_EPS  # = 0.4 + 0.02 + 0.001 = 0.421
    # Goal marker sits flush on the table surface
    goal_marker_z = TABLE_TOP_Z + GOAL_MARKER_THICKNESS / 2 + Z_EPS

    return SceneProfile(
        name="rl_push",
        table_pos=TABLE_POS,
        table_dims=TABLE_DIMS,
        table_top_z=TABLE_TOP_Z,
        robot_base_pos=ROBOT_BASE_POS,
        robot_base_quat=ROBOT_BASE_QUAT,
        obj_xy_low=OBJ_XY_LOW,
        obj_xy_high=OBJ_XY_HIGH,
        obj_yaw_range=OBJ_YAW_RANGE,
        goal_xy_low=GOAL_XY_LOW,
        goal_xy_high=GOAL_XY_HIGH,
        obj_z=obj_z,
        obj_bounds_low=OBJ_BOUNDS_LOW,
        obj_bounds_high=OBJ_BOUNDS_HIGH,
        cube_size=CUBE_SIZE,
        z_eps=Z_EPS,
        goal_marker_size=GOAL_MARKER_SIZE,
        goal_marker_thickness=GOAL_MARKER_THICKNESS,
        goal_marker_z=goal_marker_z,
        goal_marker_color=GOAL_MARKER_COLOR,
    )


# ── Object randomization parameters ──

@dataclass
class ObjectRandomizer:
    """Object randomization strategy configuration.

    .. deprecated::
        Superseded by :mod:`model.domain_randomization`, which exposes each
        randomized quantity as a registered :class:`IDomainRandomizer` and is
        configured through :class:`DomainRandomizationConfig`. Kept only for
        backwards compatibility with external callers.

    Each randomization can be toggled independently. Provides an interface to
    query per-environment randomization parameters.
    Future extension: loading random objects (different URDF/Mesh) can be added
    as a new randomization strategy.

    Size randomization:
      - Uses Genesis' heterogeneous-entity mechanism to assign a differently
        sized Box variant to each environment at build time.
      - Scale factor is sampled from a uniform distribution
        [size_scale_low, size_scale_high].
      - Size is fixed at build time and does not change at runtime
        (remains constant across episodes).

    Mass randomization:
      - Uses set_links_inertial_mass() per-env in _reset_idx.
      - Re-sampled on every episode reset.

    Friction randomization:
      - Uses set_friction_ratio() per-env in _reset_idx to set a ratio.
      - Actual friction = base_friction × ratio; must stay within [0.01, 5.0].
      - Re-sampled on every episode reset.
    """
    # Size randomization
    size_enabled: bool = False
    size_scale_low: float = 0.8       # lower bound of scale factor
    size_scale_high: float = 2.5      # upper bound of scale factor

    # Mass randomization
    mass_enabled: bool = False
    mass_low: float = 0.05            # kg
    mass_high: float = 0.4            # kg

    # Friction randomization
    friction_enabled: bool = False
    friction_low: float = 0.3         # lower bound of absolute friction
    friction_high: float = 1.0        # upper bound of absolute friction
    base_friction: float = 0.5        # base friction coefficient (for set_friction)

    # Base object size (scaling reference)
    base_size: Tuple[float, float, float] = CUBE_SIZE

    name: str = ""

    def sample_sizes(self, n_envs: int) -> List[Tuple[float, float, float]]:
        """Sample object sizes for n_envs environments (returns the size
        parameter used by the morph list).

        When size_enabled=False, all environments use base_size.
        When size_enabled=True, the scale factor is sampled from a uniform
        distribution.
        """
        if not self.size_enabled:
            return [self.base_size] * n_envs
        scales = np.random.uniform(self.size_scale_low, self.size_scale_high, size=(n_envs,))
        return [
            (float(s * self.base_size[0]),
             float(s * self.base_size[1]),
             float(s * self.base_size[2]))
            for s in scales
        ]

    def sample_masses(self, n_envs: int, device: torch.device) -> Optional[torch.Tensor]:
        """Sample masses for n_envs environments. Returns a tensor of shape
        (n_envs,).

        When mass_enabled=False, returns None.
        """
        if not self.mass_enabled:
            return None
        return torch.zeros(n_envs, device=device).uniform_(self.mass_low, self.mass_high)

    def sample_friction_ratios(self, n_envs: int, device: torch.device) -> Optional[torch.Tensor]:
        """Sample friction ratios for n_envs environments. Returns a tensor of
        shape (n_envs,).

        ratio = sampled_friction / base_friction
        When friction_enabled=False, returns None.
        """
        if not self.friction_enabled:
            return None
        frictions = torch.zeros(n_envs, device=device).uniform_(self.friction_low, self.friction_high)
        return frictions / self.base_friction

    def get_max_obj_z(self, table_top_z: float, z_eps: float) -> float:
        """计算所有可能尺寸下的最大物体Z高度（确保物体不穿透桌面）。

        使用最大缩放系数计算，确保所有变体都在桌面上方。
        """
        max_scale = self.size_scale_high if self.size_enabled else 1.0
        max_half_z = self.base_size[2] * max_scale / 2
        return table_top_z + max_half_z + z_eps


def get_rl_push_object_randomizer() -> ObjectRandomizer:
    """Get the object-randomization config for the RL pushing task
    (all randomizations disabled by default)."""
    return ObjectRandomizer(
        name="rl_push",
        size_enabled=False,
        mass_enabled=False,
        friction_enabled=False,
        base_size=CUBE_SIZE,
    )


# ── Task success-criterion parameters ──

@dataclass
class SuccessConfig:
    """Task success criterion configuration.

    Success conditions:
      1. Object-to-goal XY distance < dist_threshold
      2. Object XY linear velocity < vel_threshold (optionally enabled)
      3. ICP registration error < icp_threshold (optionally enabled)

    Attributes:
        dist_threshold: distance threshold (m)
        vel_enabled: whether to enable the velocity constraint
        vel_threshold: linear velocity threshold (m/s), XY plane only
        icp_enabled: whether to enable the ICP registration termination
        icp_threshold: ICP registration error threshold (m)
        name: configuration name
    """
    dist_threshold: float = 0.05
    vel_enabled: bool = True
    vel_threshold: float = 0.01
    icp_enabled: bool = False
    icp_threshold: float = 0.01
    name: str = ""


def get_rl_push_success_config() -> SuccessConfig:
    """Get the success-criterion config for the RL pushing task."""
    return SuccessConfig(
        name="rl_push",
        dist_threshold=0.05,
        vel_enabled=True,
        vel_threshold=0.03,
        icp_enabled=True,
        icp_threshold=0.01,
    )


def get_full_config(urdf_path: str = "") -> Dict[str, Any]:
    """Get the complete robot configuration dict (compatible with the
    create_config interface in model/factories.py)."""
    return {
        "urdf_path": urdf_path or URDF_PATH,
        "robot_name": "yhrg_s1",
        "joint_names": JOINT_NAMES,
        "total_dof": TOTAL_DOF,
        "fixed_base": True,
        "base_link_name": BASE_LINK_NAME,
        "joint_limits": {
            "lower": JOINT_LIMITS_LOWER.copy(),
            "upper": JOINT_LIMITS_UPPER.copy(),
        },
        "control_gains": {
            "kp": CONTROL_KP.copy(),
            "kv": CONTROL_KV.copy(),
        },
        "force_limits": {
            "lower": FORCE_LIMITS_LOWER.copy(),
            "upper": FORCE_LIMITS_UPPER.copy(),
        },
        "default_positions": {
            "neutral": NEUTRAL_POSITION.copy(),
            "home": HOME_POSITION.copy(),
            "rl_push": RL_PUSH_DEFAULT_POSITION.copy(),
        },
    }
