"""
Loader Interfaces and Implementations for Model Layer.

This module provides loader components using design patterns:
- Abstract Factory Pattern: ISceneLoaderFactory, IRobotLoaderFactory, IObjectLoaderFactory
- Simple Factory Pattern: LoaderFactoryRegistry for concrete loader creation

The loaders are responsible for:
- Loading scenes, robots, and objects into the simulation
- Managing asset configurations
- Providing standardized loading interfaces
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from config.scene_config import (
    SceneConfig,
    RobotConfig,
    DesktopConfig,
    TargetObjectConfig,
)
from model.state import RobotState, SceneState
from model.registry import Registry

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """
    Result of a loading operation.
    
    Attributes:
        success: Whether the loading was successful.
        entity: The loaded entity (if successful).
        error_message: Error message (if failed).
        metadata: Additional metadata about the loaded entity.
    """
    success: bool
    entity: Any = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, entity: Any, **metadata) -> "LoadResult":
        """Create a successful result."""
        return cls(success=True, entity=entity, metadata=metadata)
    
    @classmethod
    def error(cls, message: str, **metadata) -> "LoadResult":
        """Create an error result."""
        return cls(success=False, error_message=message, metadata=metadata)


class ISceneLoaderFactory(ABC):
    """
    Abstract factory interface for scene loaders.
    
    Defines the contract for creating scene loaders that can load
    complete simulation scenes with different simulator backends.
    """
    
    @abstractmethod
    def create_loader(self, config: SceneConfig) -> "ISceneLoader":
        """
        Create a scene loader instance.
        
        Args:
            config: Scene configuration.
            
        Returns:
            ISceneLoader instance.
        """
        pass
    
    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.
        
        Returns:
            String identifier (e.g., "genesis", "pybullet").
        """
        pass


class IRobotLoaderFactory(ABC):
    """
    Abstract factory interface for robot loaders.
    
    Defines the contract for creating robot loaders that can load
    robot models from URDF or other model formats.
    """
    
    @abstractmethod
    def create_loader(self, config: RobotConfig) -> "IRobotLoader":
        """
        Create a robot loader instance.
        
        Args:
            config: Robot configuration.
            
        Returns:
            IRobotLoader instance.
        """
        pass
    
    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.
        
        Returns:
            String identifier (e.g., "urdf", "mjcf").
        """
        pass


class IObjectLoaderFactory(ABC):
    """
    Abstract factory interface for object loaders.
    
    Defines the contract for creating object loaders that can load
    target objects, desktops, and other scene objects.
    """
    
    @abstractmethod
    def create_loader(self, config: Any) -> "IObjectLoader":
        """
        Create an object loader instance.
        
        Args:
            config: Object configuration.
            
        Returns:
            IObjectLoader instance.
        """
        pass
    
    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.
        
        Returns:
            String identifier (e.g., "urdf", "mesh").
        """
        pass


class ISceneLoader(ABC):
    """
    Abstract interface for scene loading operations.
    
    Defines the contract for loading complete simulation scenes.
    """
    
    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the scene.
        
        Args:
            context: Context containing simulator reference.
            
        Returns:
            LoadResult with scene entity.
        """
        pass


class IRobotLoader(ABC):
    """
    Abstract interface for robot loading operations.
    
    Defines the contract for loading robot models.
    """
    
    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the robot.
        
        Args:
            context: Context containing scene reference.
            
        Returns:
            LoadResult with robot entity.
        """
        pass
    
    @abstractmethod
    def get_robot_state(self) -> RobotState:
        """
        Get the robot state after loading.
        
        Returns:
            RobotState instance.
        """
        pass


class IObjectLoader(ABC):
    """
    Abstract interface for object loading operations.
    
    Defines the contract for loading scene objects.
    """
    
    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the object.
        
        Args:
            context: Context containing scene reference.
            
        Returns:
            LoadResult with object entity.
        """
        pass


class GenesisSceneLoader(ISceneLoader):
    """
    Scene loader implementation for Genesis simulator.
    
    Loads complete simulation scenes including ground plane,
    lighting, and camera configuration.
    """
    
    def __init__(self, config: SceneConfig):
        """
        Initialize the Genesis scene loader.
        
        Args:
            config: Scene configuration.
        """
        self._config = config
        self._scene = None
        self._gs = None
        logger.info("GenesisSceneLoader initialized")
    
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the Genesis scene.
        
        Args:
            context: Context containing 'gs' (Genesis module).
            
        Returns:
            LoadResult with scene entity.
        """
        self._gs = context.get("gs")
        if self._gs is None:
            return LoadResult.error("Genesis module not available in context")
        
        try:
            camera_config = self._config.camera
            ground_config = self._config.ground
            
            self._scene = self._gs.Scene(
                viewer_options=self._gs.options.ViewerOptions(
                    camera_pos=camera_config.position,
                    camera_lookat=camera_config.lookat,
                    camera_fov=camera_config.fov,
                    max_FPS=60,
                ),
                sim_options=self._gs.options.SimOptions(
                    dt=0.01,
                ),
                show_viewer=context.get("show_viewer", True),
            )
            
            if ground_config.enabled:
                plane = self._scene.add_entity(self._gs.morphs.Plane())
                context["plane"] = plane
            
            context["scene"] = self._scene
            context["scene_config"] = self._config
            
            logger.info("Genesis scene loaded successfully")
            return LoadResult.ok(
                self._scene,
                ground_enabled=ground_config.enabled,
                camera_position=camera_config.position,
            )
            
        except Exception as e:
            logger.error(f"Failed to load Genesis scene: {e}")
            return LoadResult.error(str(e))
    
    def get_scene(self):
        """Get the loaded scene."""
        return self._scene


class GenesisRobotLoader(IRobotLoader):
    """
    Robot loader implementation for Genesis simulator.
    
    Loads robot models from URDF files and initializes joint states.
    """
    
    def __init__(self, config: RobotConfig, robot_state: Optional[RobotState] = None):
        """
        Initialize the Genesis robot loader.
        
        Args:
            config: Robot configuration.
            robot_state: Optional pre-configured robot state.
        """
        self._config = config
        self._robot_state = robot_state or RobotState(
            name=config.name,
            urdf_path=config.urdf_path,
            fixed_base=config.fixed_base,
        )
        self._robot = None
        self._motor_dof_idx = []
        logger.info(f"GenesisRobotLoader initialized for: {config.urdf_path}")
    
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the robot into the scene.
        
        Args:
            context: Context containing 'scene' and 'gs'.
            
        Returns:
            LoadResult with robot entity.
        """
        scene = context.get("scene")
        gs = context.get("gs")
        
        if scene is None:
            return LoadResult.error("Scene not available in context")
        if gs is None:
            return LoadResult.error("Genesis module not available in context")
        
        urdf_path = self._config.urdf_path
        if not os.path.exists(urdf_path):
            return LoadResult.error(f"URDF file not found: {urdf_path}")
        
        try:
            self._robot = scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    pos=self._config.base_position,
                    quat=self._config.base_orientation,
                    fixed=self._config.fixed_base,
                ),
            )
            
            robot_config = context.get("robot_config", {})
            joint_names = robot_config.get("joint_names", ())
            
            self._motor_dof_idx = []
            for name in joint_names:
                try:
                    joint = self._robot.get_joint(name)
                    self._motor_dof_idx.append(joint.dofs_idx_local[0])
                except Exception as e:
                    logger.warning(f"Could not find joint '{name}': {e}")
            
            self._robot_state.initialize_joints(joint_names)
            
            joint_limits = robot_config.get("joint_limits", {})
            self._robot_state.joint_limits_lower = np.array(
                joint_limits.get("lower", np.zeros(len(joint_names)))
            )
            self._robot_state.joint_limits_upper = np.array(
                joint_limits.get("upper", np.zeros(len(joint_names)))
            )
            
            control_gains = robot_config.get("control_gains", {})
            self._robot_state.control_gains_kp = np.array(
                control_gains.get("kp", np.ones(len(joint_names)))
            )
            self._robot_state.control_gains_kv = np.array(
                control_gains.get("kv", np.ones(len(joint_names)))
            )
            
            force_limits = robot_config.get("force_limits", {})
            self._robot_state.force_limits_lower = np.array(
                force_limits.get("lower", np.full(len(joint_names), -10.0))
            )
            self._robot_state.force_limits_upper = np.array(
                force_limits.get("upper", np.full(len(joint_names), 10.0))
            )
            
            context["robot"] = self._robot
            context["motor_dof_idx"] = self._motor_dof_idx
            context["robot_state"] = self._robot_state
            
            logger.info(f"Robot loaded successfully with {len(self._motor_dof_idx)} joints")
            return LoadResult.ok(
                self._robot,
                num_joints=len(self._motor_dof_idx),
                joint_names=list(joint_names),
            )
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return LoadResult.error(str(e))
    
    def get_robot_state(self) -> RobotState:
        """Get the robot state."""
        return self._robot_state
    
    def get_motor_dof_indices(self) -> List[int]:
        """Get the motor DOF indices."""
        return self._motor_dof_idx


class GenesisObjectLoader(IObjectLoader):
    """
    Object loader implementation for Genesis simulator.
    
    Loads scene objects like desktops and target objects.
    """
    
    def __init__(self, config: Any, object_type: str = "object"):
        """
        Initialize the Genesis object loader.
        
        Args:
            config: Object configuration (DesktopConfig or TargetObjectConfig).
            object_type: Type identifier for the object.
        """
        self._config = config
        self._object_type = object_type
        self._entity = None
        logger.info(f"GenesisObjectLoader initialized for {object_type}")
    
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the object into the scene.
        
        Args:
            context: Context containing 'scene' and 'gs'.
            
        Returns:
            LoadResult with object entity.
        """
        scene = context.get("scene")
        gs = context.get("gs")
        
        if scene is None:
            return LoadResult.error("Scene not available in context")
        if gs is None:
            return LoadResult.error("Genesis module not available in context")
        
        urdf_path = getattr(self._config, "urdf_path", None)
        if urdf_path is None:
            return LoadResult.error("Configuration does not have urdf_path")
        
        if not os.path.exists(urdf_path):
            logger.warning(f"Object URDF not found: {urdf_path}")
            return LoadResult.error(f"URDF file not found: {urdf_path}")
        
        try:
            position = getattr(self._config, "position", (0, 0, 0))
            orientation = getattr(self._config, "orientation", (1, 0, 0, 0))
            fixed = getattr(self._config, "fixed", False)
            
            self._entity = scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    pos=position,
                    quat=orientation,
                    fixed=fixed,
                ),
            )
            
            object_name = getattr(self._config, "name", self._object_type)
            context[f"{object_name}_entity"] = self._entity
            
            logger.info(f"Object '{object_name}' loaded at position: {position}")
            return LoadResult.ok(
                self._entity,
                name=object_name,
                position=position,
            )
            
        except Exception as e:
            logger.error(f"Failed to load object: {e}")
            return LoadResult.error(str(e))


class GenesisSceneLoaderFactory(ISceneLoaderFactory):
    """
    Factory for creating Genesis scene loaders.
    
    Implements the Abstract Factory pattern for scene loader creation.
    """
    
    def create_loader(self, config: SceneConfig) -> ISceneLoader:
        """
        Create a Genesis scene loader.
        
        Args:
            config: Scene configuration.
            
        Returns:
            GenesisSceneLoader instance.
        """
        return GenesisSceneLoader(config)
    
    def get_loader_type(self) -> str:
        return "genesis"


class GenesisRobotLoaderFactory(IRobotLoaderFactory):
    """
    Factory for creating Genesis robot loaders.
    
    Implements the Abstract Factory pattern for robot loader creation.
    """
    
    def create_loader(self, config: RobotConfig) -> IRobotLoader:
        """
        Create a Genesis robot loader.
        
        Args:
            config: Robot configuration.
            
        Returns:
            GenesisRobotLoader instance.
        """
        return GenesisRobotLoader(config)
    
    def get_loader_type(self) -> str:
        return "genesis_urdf"


class GenesisObjectLoaderFactory(IObjectLoaderFactory):
    """
    Factory for creating Genesis object loaders.
    
    Implements the Abstract Factory pattern for object loader creation.
    """
    
    def create_loader(self, config: Any) -> IObjectLoader:
        """
        Create a Genesis object loader.
        
        Args:
            config: Object configuration.
            
        Returns:
            GenesisObjectLoader instance.
        """
        object_type = getattr(config, "name", "object")
        return GenesisObjectLoader(config, object_type)
    
    def get_loader_type(self) -> str:
        return "genesis_urdf"


# Module-level registries for loader factories
_scene_registry = Registry[ISceneLoaderFactory]("scene_loaders")
_robot_registry = Registry[IRobotLoaderFactory]("robot_loaders")
_object_registry = Registry[IObjectLoaderFactory]("object_loaders")

_scene_registry.register("genesis", GenesisSceneLoaderFactory)
_robot_registry.register("genesis_urdf", GenesisRobotLoaderFactory)
_object_registry.register("genesis_urdf", GenesisObjectLoaderFactory)


class LoaderFactoryRegistry:
    """
    Registry for loader factories, backed by model.registry.Registry.
    
    Provides a centralized registry for creating loaders of different types,
    enabling easy extension to new simulator backends.
    """
    
    @classmethod
    def register_scene_factory(cls, name: str, factory_class: Type[ISceneLoaderFactory]) -> None:
        _scene_registry.register(name, factory_class)
    
    @classmethod
    def register_robot_factory(cls, name: str, factory_class: Type[IRobotLoaderFactory]) -> None:
        _robot_registry.register(name, factory_class)
    
    @classmethod
    def register_object_factory(cls, name: str, factory_class: Type[IObjectLoaderFactory]) -> None:
        _object_registry.register(name, factory_class)
    
    @classmethod
    def create_scene_loader(cls, name: str, config: SceneConfig) -> ISceneLoader:
        factory = _scene_registry.create(name)
        return factory.create_loader(config)
    
    @classmethod
    def create_robot_loader(cls, name: str, config: RobotConfig) -> IRobotLoader:
        factory = _robot_registry.create(name)
        return factory.create_loader(config)
    
    @classmethod
    def create_object_loader(cls, name: str, config: Any) -> IObjectLoader:
        factory = _object_registry.create(name)
        return factory.create_loader(config)
