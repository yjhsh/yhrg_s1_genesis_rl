"""
Simulator Proxy Interface and Implementations for View Layer.

This module provides simulator communication using the Proxy Pattern:
- ISimulatorProxy: Abstract interface for simulator communication
- GenesisSimulatorProxy: Genesis-specific implementation
- SimulatorProxyFactory: Factory for creating proxy instances

The Proxy pattern enables:
- Decoupled communication between controller and simulator
- Easy replacement of simulator backends
- Centralized simulator-specific logic
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from model.registry import Registry

logger = logging.getLogger(__name__)


class ISimulatorProxy(ABC):
    """
    Abstract interface for simulator communication.
    
    This interface defines the contract for communicating with different
    physics simulators, enabling decoupled and interchangeable backends.
    
    The Proxy pattern allows the Controller layer to interact with the
    simulator without knowing the specific implementation details.
    """
    
    @abstractmethod
    def initialize(self, backend: str = "gpu") -> bool:
        """
        Initialize the simulator.
        
        Args:
            backend: Computation backend ("gpu" or "cpu").
            
        Returns:
            True if initialization successful.
        """
        pass
    
    @abstractmethod
    def create_scene(self, **options) -> Any:
        """
        Create a simulation scene.
        
        Args:
            **options: Scene configuration options.
            
        Returns:
            Scene entity.
        """
        pass
    
    @abstractmethod
    def load_urdf(self, scene: Any, urdf_path: str, **options) -> Any:
        """
        Load a URDF model into the scene.
        
        Args:
            scene: The scene entity.
            urdf_path: Path to the URDF file.
            **options: Loading options (position, orientation, fixed, etc.).
            
        Returns:
            Loaded entity.
        """
        pass
    
    @abstractmethod
    def add_plane(self, scene: Any) -> Any:
        """
        Add a ground plane to the scene.
        
        Args:
            scene: The scene entity.
            
        Returns:
            Plane entity.
        """
        pass
    
    @abstractmethod
    def build_scene(self, scene: Any) -> None:
        """
        Build the scene for simulation.
        
        Args:
            scene: The scene entity.
        """
        pass
    
    @abstractmethod
    def step(self, scene: Any) -> None:
        """
        Step the simulation forward.
        
        Args:
            scene: The scene entity.
        """
        pass
    
    @abstractmethod
    def get_joint_positions(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint positions.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint positions.
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint velocities.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint velocities.
        """
        pass
    
    @abstractmethod
    def get_joint_efforts(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint efforts.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint efforts.
        """
        pass
    
    @abstractmethod
    def set_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint positions directly.
        
        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def control_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints to target positions.
        
        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target velocities.
        
        Args:
            robot: The robot entity.
            velocities: Target velocities.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def control_joint_efforts(self, robot: Any, efforts: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target efforts.
        
        Args:
            robot: The robot entity.
            efforts: Target efforts.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def set_joint_kp(self, robot: Any, kp: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set position control gains.
        
        Args:
            robot: The robot entity.
            kp: Position gains.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def set_joint_kv(self, robot: Any, kv: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set velocity control gains.
        
        Args:
            robot: The robot entity.
            kv: Velocity gains.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def set_joint_force_range(self, robot: Any, lower: np.ndarray, upper: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint force limits.
        
        Args:
            robot: The robot entity.
            lower: Lower force limits.
            upper: Upper force limits.
            joint_indices: List of joint DOF indices.
        """
        pass
    
    @abstractmethod
    def get_joint(self, robot: Any, joint_name: str) -> Any:
        """
        Get a joint by name.
        
        Args:
            robot: The robot entity.
            joint_name: Name of the joint.
            
        Returns:
            Joint entity.
        """
        pass
    
    @abstractmethod
    def get_control_force(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current control forces.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of control forces.
        """
        pass
    
    @abstractmethod
    def get_simulator_type(self) -> str:
        """
        Return the simulator type identifier.
        
        Returns:
            String identifier (e.g., "genesis", "pybullet").
        """
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the simulator is initialized.
        
        Returns:
            True if initialized.
        """
        pass
    
    @abstractmethod
    def get_simulator_module(self) -> Any:
        """
        Get the underlying simulator module.
        
        Returns:
            The simulator module for advanced usage.
        """
        pass


class GenesisSimulatorProxy(ISimulatorProxy):
    """
    Genesis-specific simulator proxy implementation.
    
    This class implements the ISimulatorProxy interface for the Genesis
    physics simulator, providing a clean abstraction layer for all
    Genesis-specific operations.
    
    The Proxy pattern enables:
    - Decoupling from Genesis-specific APIs
    - Centralized Genesis-specific logic
    - Easy testing with mock implementations
    """
    
    def __init__(self):
        """Initialize the Genesis simulator proxy."""
        self._gs = None
        self._initialized = False
        self._backend = "gpu"
        logger.info("GenesisSimulatorProxy created")
    
    def initialize(self, backend: str = "gpu") -> bool:
        """
        Initialize the Genesis simulator.
        
        Args:
            backend: Computation backend ("gpu" or "cpu").
            
        Returns:
            True if initialization successful.
        """
        if self._initialized:
            logger.warning("Genesis already initialized")
            return True
        
        try:
            import genesis as gs
            
            self._backend = backend.lower()
            backend_enum = gs.gpu if self._backend == "gpu" else gs.cpu
            
            logger.info(f"Initializing Genesis with {self._backend.upper()} backend...")
            gs.init(backend=backend_enum)
            
            self._gs = gs
            self._initialized = True
            logger.info("Genesis simulator initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Genesis not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Genesis: {e}")
            return False
    
    def create_scene(self, **options) -> Any:
        """
        Create a Genesis simulation scene.
        
        Args:
            **options: Scene options including:
                - camera_pos: Camera position tuple
                - camera_lookat: Camera look-at target
                - camera_fov: Field of view
                - max_fps: Maximum FPS
                - dt: Simulation time step
                - show_viewer: Whether to show viewer
                
        Returns:
            Genesis Scene entity.
        """
        if not self._initialized:
            raise RuntimeError("Simulator not initialized")
        
        viewer_options = self._gs.options.ViewerOptions(
            camera_pos=options.get("camera_pos", (0.0, -3.5, 2.5)),
            camera_lookat=options.get("camera_lookat", (0.0, 0.0, 0.5)),
            camera_fov=options.get("camera_fov", 30.0),
            max_FPS=options.get("max_fps", 60),
        )
        
        sim_options = self._gs.options.SimOptions(
            dt=options.get("dt", 0.01),
        )
        
        scene = self._gs.Scene(
            viewer_options=viewer_options,
            sim_options=sim_options,
            show_viewer=options.get("show_viewer", True),
        )
        
        logger.info("Genesis scene created")
        return scene
    
    def load_urdf(self, scene: Any, urdf_path: str, **options) -> Any:
        """
        Load a URDF model into the scene.
        
        Args:
            scene: The Genesis scene.
            urdf_path: Path to URDF file.
            **options: Loading options (pos, quat, fixed).
            
        Returns:
            Entity loaded from URDF.
        """
        entity = scene.add_entity(
            self._gs.morphs.URDF(
                file=urdf_path,
                pos=options.get("pos", (0.0, 0.0, 0.0)),
                quat=options.get("quat", (1.0, 0.0, 0.0, 0.0)),
                fixed=options.get("fixed", False),
            ),
        )
        logger.debug(f"URDF loaded: {urdf_path}")
        return entity
    
    def add_plane(self, scene: Any) -> Any:
        """
        Add a ground plane to the scene.
        
        Args:
            scene: The Genesis scene.
            
        Returns:
            Plane entity.
        """
        plane = scene.add_entity(self._gs.morphs.Plane())
        logger.debug("Ground plane added")
        return plane
    
    def build_scene(self, scene: Any) -> None:
        """
        Build the scene for simulation.
        
        Args:
            scene: The Genesis scene.
        """
        scene.build()
        logger.info("Scene built")
    
    def step(self, scene: Any) -> None:
        """
        Step the simulation forward.
        
        Args:
            scene: The Genesis scene.
        """
        scene.step()
    
    def get_joint_positions(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint positions.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint positions.
        """
        return robot.get_dofs_position(joint_indices)
    
    def get_joint_velocities(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint velocities.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint velocities.
        """
        return robot.get_dofs_velocity(joint_indices)
    
    def get_joint_efforts(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint efforts.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of joint efforts.
        """
        return robot.get_dofs_force(joint_indices)
    
    def set_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint positions directly.
        
        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        robot.set_dofs_position(positions, joint_indices)
    
    def control_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints to target positions.
        
        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        robot.control_dofs_position(positions, joint_indices)
    
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target velocities.
        
        Args:
            robot: The robot entity.
            velocities: Target velocities.
            joint_indices: List of joint DOF indices.
        """
        robot.control_dofs_velocity(velocities, joint_indices)
    
    def control_joint_efforts(self, robot: Any, efforts: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target efforts.
        
        Args:
            robot: The robot entity.
            efforts: Target efforts.
            joint_indices: List of joint DOF indices.
        """
        robot.control_dofs_force(efforts, joint_indices)
    
    def set_joint_kp(self, robot: Any, kp: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set position control gains.
        
        Args:
            robot: The robot entity.
            kp: Position gains.
            joint_indices: List of joint DOF indices.
        """
        robot.set_dofs_kp(kp=kp, dofs_idx_local=joint_indices)
    
    def set_joint_kv(self, robot: Any, kv: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set velocity control gains.
        
        Args:
            robot: The robot entity.
            kv: Velocity gains.
            joint_indices: List of joint DOF indices.
        """
        robot.set_dofs_kv(kv=kv, dofs_idx_local=joint_indices)
    
    def set_joint_force_range(self, robot: Any, lower: np.ndarray, upper: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint force limits.
        
        Args:
            robot: The robot entity.
            lower: Lower force limits.
            upper: Upper force limits.
            joint_indices: List of joint DOF indices.
        """
        robot.set_dofs_force_range(lower=lower, upper=upper, dofs_idx_local=joint_indices)
    
    def get_joint(self, robot: Any, joint_name: str) -> Any:
        """
        Get a joint by name.
        
        Args:
            robot: The robot entity.
            joint_name: Name of the joint.
            
        Returns:
            Joint entity.
        """
        return robot.get_joint(joint_name)
    
    def get_control_force(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current control forces.
        
        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.
            
        Returns:
            Array of control forces.
        """
        return robot.get_dofs_control_force(joint_indices)
    
    def get_simulator_type(self) -> str:
        return "genesis"
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def get_simulator_module(self) -> Any:
        return self._gs


# Module-level registry instance for simulator proxies
_proxy_registry = Registry[ISimulatorProxy]("simulator_proxies")
_proxy_registry.register("genesis", GenesisSimulatorProxy)


class SimulatorProxyFactory:
    """
    Factory for creating simulator proxy instances, backed by model.registry.Registry.
    """
    
    @classmethod
    def register(cls, name: str, proxy_class: Type[ISimulatorProxy]) -> None:
        _proxy_registry.register(name, proxy_class)
    
    @classmethod
    def create(cls, name: str = "genesis") -> ISimulatorProxy:
        return _proxy_registry.create(name)
    
    @classmethod
    def get_available_simulators(cls) -> List[str]:
        return _proxy_registry.get_available()
