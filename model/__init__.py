"""
Model Layer for Genesis Simulation.

This package contains the Model components following MVC architecture:
- Robot configuration (Single Source of Truth: model.robot_config)
- State management classes
- Abstract factory interfaces for loaders
- Concrete loader implementations
- Registry base class
- Business logic and data structures

The Model layer is responsible for:
- Managing simulation state and robot state
- Loading scenes, objects, and robots
- Providing data access interfaces
- Core business logic without UI dependencies
"""

from model.state import RobotState, SimulationState, SceneState
from model.robot_config import (
    URDF_PATH,
    JOINT_NAMES,
    ARM_DOF,
    GRIPPER_DOF,
    TOTAL_DOF,
    CONTROL_KP,
    CONTROL_KV,
    FORCE_LIMITS_LOWER,
    FORCE_LIMITS_UPPER,
    NEUTRAL_POSITION,
    HOME_POSITION,
    RL_PUSH_DEFAULT_POSITION,
    RobotProfile,
    get_interactive_profile,
    get_rl_push_profile,
    get_full_config,
    SuccessConfig,
    get_rl_push_success_config,
)
from model.registry import Registry
from model.loaders import (
    ISceneLoaderFactory,
    IRobotLoaderFactory,
    IObjectLoaderFactory,
    GenesisSceneLoaderFactory,
    GenesisRobotLoaderFactory,
    GenesisObjectLoaderFactory,
    LoaderFactoryRegistry,
)
from model.factories import (
    RobotConfigFactory,
    YHRGS1RobotConfigFactory,
)
from model.domain_randomization import (
    Phase,
    RandomizationContext,
    IDomainRandomizer,
    RANDOMIZER_REGISTRY,
    randomizer,
    randomize,
    SizeRandomizer,
    MassRandomizer,
    FrictionRandomizer,
    DomainRandomizationConfig,
    DomainRandomizerManager,
    get_rl_push_dr_config,
)
from model.pointcloud import (
    ISamplingStrategy,
    UniformSamplingStrategy,
    PointCloudConfig,
    PointCloudProcessor,
    get_rl_push_pointcloud_config,
    box_surface_sample,
    farthest_point_sample,
    icp_align,
    transform_pcd_by_pose,
    generate_goal_point_cloud,
)

__all__ = [
    # State
    "RobotState",
    "SimulationState",
    "SceneState",
    # Robot Config (Single Source of Truth)
    "URDF_PATH",
    "JOINT_NAMES",
    "ARM_DOF",
    "GRIPPER_DOF",
    "TOTAL_DOF",
    "CONTROL_KP",
    "CONTROL_KV",
    "FORCE_LIMITS_LOWER",
    "FORCE_LIMITS_UPPER",
    "NEUTRAL_POSITION",
    "HOME_POSITION",
    "RL_PUSH_DEFAULT_POSITION",
    "RobotProfile",
    "get_interactive_profile",
    "get_rl_push_profile",
    "get_full_config",
    # Success Config
    "SuccessConfig",
    "get_rl_push_success_config",
    # Registry
    "Registry",
    # Loaders
    "ISceneLoaderFactory",
    "IRobotLoaderFactory",
    "IObjectLoaderFactory",
    "GenesisSceneLoaderFactory",
    "GenesisRobotLoaderFactory",
    "GenesisObjectLoaderFactory",
    "LoaderFactoryRegistry",
    # Factories
    "RobotConfigFactory",
    "YHRGS1RobotConfigFactory",
    # Domain Randomization
    "Phase",
    "RandomizationContext",
    "IDomainRandomizer",
    "RANDOMIZER_REGISTRY",
    "randomizer",
    "randomize",
    "SizeRandomizer",
    "MassRandomizer",
    "FrictionRandomizer",
    "DomainRandomizationConfig",
    "DomainRandomizerManager",
    "get_rl_push_dr_config",
    # Point Cloud
    "ISamplingStrategy",
    "UniformSamplingStrategy",
    "PointCloudConfig",
    "PointCloudProcessor",
    "get_rl_push_pointcloud_config",
    "box_surface_sample",
    "farthest_point_sample",
    "icp_align",
    "transform_pcd_by_pose",
    "generate_goal_point_cloud",
]
