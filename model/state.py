"""
State Management Classes for Model Layer.

This module provides state management classes following the MVC pattern:
- RobotState: Manages robot joint states and configurations
- SimulationState: Manages overall simulation state
- SceneState: Manages scene configuration and entities

These classes are pure data structures with no UI dependencies,
following the Single Responsibility Principle.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


class SimulationStatus(Enum):
    """Enumeration of simulation status states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class JointState:
    """
    Data class representing a single joint's state.
    
    Attributes:
        name: Joint name identifier.
        position: Current position in radians or meters.
        velocity: Current velocity.
        effort: Current effort/force.
        target_position: Target position for control.
    """
    name: str
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    target_position: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "position": self.position,
            "velocity": self.velocity,
            "effort": self.effort,
            "target_position": self.target_position,
        }


@dataclass
class RobotState:
    """
    Manages the complete state of a robot.
    
    This class encapsulates all robot-related state information including
    joint states, control parameters, and configuration data. It serves
    as the single source of truth for robot state in the Model layer.
    
    Attributes:
        name: Robot identifier.
        joint_states: Dictionary mapping joint names to JointState objects.
        joint_names: Ordered list of joint names.
        joint_limits_lower: Lower position limits for each joint.
        joint_limits_upper: Upper position limits for each joint.
        control_gains_kp: Position control gains.
        control_gains_kv: Velocity control gains.
        force_limits_lower: Lower force limits.
        force_limits_upper: Upper force limits.
        fixed_base: Whether the robot base is fixed.
        urdf_path: Path to the URDF file.
    """
    name: str = "robot"
    joint_states: Dict[str, JointState] = field(default_factory=dict)
    joint_names: Tuple[str, ...] = field(default_factory=tuple)
    joint_limits_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    control_gains_kp: np.ndarray = field(default_factory=lambda: np.array([]))
    control_gains_kv: np.ndarray = field(default_factory=lambda: np.array([]))
    force_limits_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    force_limits_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    fixed_base: bool = True
    urdf_path: str = ""
    _timestamp: float = field(default_factory=time.time)
    
    @property
    def num_joints(self) -> int:
        """Return the number of joints."""
        return len(self.joint_names)
    
    @property
    def positions(self) -> np.ndarray:
        """Return current positions as numpy array."""
        return np.array([self.joint_states[name].position for name in self.joint_names])
    
    @property
    def velocities(self) -> np.ndarray:
        """Return current velocities as numpy array."""
        return np.array([self.joint_states[name].velocity for name in self.joint_names])
    
    @property
    def efforts(self) -> np.ndarray:
        """Return current efforts as numpy array."""
        return np.array([self.joint_states[name].effort for name in self.joint_names])
    
    @property
    def target_positions(self) -> np.ndarray:
        """Return target positions as numpy array."""
        return np.array([self.joint_states[name].target_position for name in self.joint_names])
    
    def update_positions(self, positions: np.ndarray) -> None:
        """
        Update all joint positions.
        
        Args:
            positions: Array of new positions.
        """
        for i, name in enumerate(self.joint_names):
            if i < len(positions):
                self.joint_states[name].position = float(positions[i])
        self._timestamp = time.time()
    
    def update_velocities(self, velocities: np.ndarray) -> None:
        """
        Update all joint velocities.
        
        Args:
            velocities: Array of new velocities.
        """
        for i, name in enumerate(self.joint_names):
            if i < len(velocities):
                self.joint_states[name].velocity = float(velocities[i])
        self._timestamp = time.time()
    
    def update_efforts(self, efforts: np.ndarray) -> None:
        """
        Update all joint efforts.
        
        Args:
            efforts: Array of new efforts.
        """
        for i, name in enumerate(self.joint_names):
            if i < len(efforts):
                self.joint_states[name].effort = float(efforts[i])
        self._timestamp = time.time()
    
    def set_target_positions(self, targets: np.ndarray) -> None:
        """
        Set target positions for all joints.
        
        Args:
            targets: Array of target positions.
        """
        for i, name in enumerate(self.joint_names):
            if i < len(targets):
                self.joint_states[name].target_position = float(targets[i])
    
    def initialize_joints(self, joint_names: Tuple[str, ...]) -> None:
        """
        Initialize joint states for the given joint names.
        
        Args:
            joint_names: Tuple of joint names in control order.
        """
        self.joint_names = joint_names
        self.joint_states = {name: JointState(name=name) for name in joint_names}
        
        num_joints = len(joint_names)
        if len(self.joint_limits_lower) != num_joints:
            self.joint_limits_lower = np.zeros(num_joints)
        if len(self.joint_limits_upper) != num_joints:
            self.joint_limits_upper = np.zeros(num_joints)
        if len(self.control_gains_kp) != num_joints:
            self.control_gains_kp = np.ones(num_joints)
        if len(self.control_gains_kv) != num_joints:
            self.control_gains_kv = np.ones(num_joints)
        if len(self.force_limits_lower) != num_joints:
            self.force_limits_lower = np.full(num_joints, -10.0)
        if len(self.force_limits_upper) != num_joints:
            self.force_limits_upper = np.full(num_joints, 10.0)
    
    def get_joint_state(self, name: str) -> Optional[JointState]:
        """
        Get the state of a specific joint.
        
        Args:
            name: Joint name.
            
        Returns:
            JointState if found, None otherwise.
        """
        return self.joint_states.get(name)
    
    def clamp_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Clamp positions to joint limits.
        
        Args:
            positions: Input positions.
            
        Returns:
            Clamped positions.
        """
        return np.clip(positions, self.joint_limits_lower, self.joint_limits_upper)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "efforts": self.efforts.tolist(),
            "target_positions": self.target_positions.tolist(),
            "joint_limits_lower": self.joint_limits_lower.tolist(),
            "joint_limits_upper": self.joint_limits_upper.tolist(),
            "fixed_base": self.fixed_base,
            "timestamp": self._timestamp,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RobotState":
        """
        Create RobotState from configuration dictionary.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Initialized RobotState instance.
        """
        state = cls(
            name=config.get("robot_name", "robot"),
            urdf_path=config.get("urdf_path", ""),
            fixed_base=config.get("fixed_base", True),
        )
        
        joint_names = config.get("joint_names", ())
        state.initialize_joints(joint_names)
        
        joint_limits = config.get("joint_limits", {})
        state.joint_limits_lower = np.array(joint_limits.get("lower", []))
        state.joint_limits_upper = np.array(joint_limits.get("upper", []))
        
        control_gains = config.get("control_gains", {})
        state.control_gains_kp = np.array(control_gains.get("kp", []))
        state.control_gains_kv = np.array(control_gains.get("kv", []))
        
        force_limits = config.get("force_limits", {})
        state.force_limits_lower = np.array(force_limits.get("lower", []))
        state.force_limits_upper = np.array(force_limits.get("upper", []))
        
        default_positions = config.get("default_positions", {})
        if "neutral" in default_positions:
            state.set_target_positions(np.array(default_positions["neutral"]))
        
        return state


@dataclass
class SceneState:
    """
    Manages the state of the simulation scene.
    
    This class encapsulates all scene-related state including
    entities, configurations, and environment settings.
    
    Attributes:
        name: Scene identifier.
        entities: Dictionary of loaded entities by name.
        ground_enabled: Whether ground plane is enabled.
        camera_position: Camera position tuple.
        camera_lookat: Camera look-at target.
        desktop_loaded: Whether desktop has been loaded.
        target_objects_loaded: List of loaded target object names.
    """
    name: str = "scene"
    entities: Dict[str, Any] = field(default_factory=dict)
    ground_enabled: bool = True
    camera_position: Tuple[float, float, float] = (0.0, -3.5, 2.5)
    camera_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_fov: float = 30.0
    desktop_loaded: bool = False
    target_objects_loaded: List[str] = field(default_factory=list)
    _timestamp: float = field(default_factory=time.time)
    
    def add_entity(self, name: str, entity: Any) -> None:
        """
        Add an entity to the scene state.
        
        Args:
            name: Entity identifier.
            entity: The entity object.
        """
        self.entities[name] = entity
        self._timestamp = time.time()
    
    def get_entity(self, name: str) -> Optional[Any]:
        """
        Get an entity by name.
        
        Args:
            name: Entity identifier.
            
        Returns:
            Entity if found, None otherwise.
        """
        return self.entities.get(name)
    
    def remove_entity(self, name: str) -> bool:
        """
        Remove an entity from the scene.
        
        Args:
            name: Entity identifier.
            
        Returns:
            True if removed, False if not found.
        """
        if name in self.entities:
            del self.entities[name]
            self._timestamp = time.time()
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "entities": list(self.entities.keys()),
            "ground_enabled": self.ground_enabled,
            "camera_position": self.camera_position,
            "camera_lookat": self.camera_lookat,
            "camera_fov": self.camera_fov,
            "desktop_loaded": self.desktop_loaded,
            "target_objects_loaded": self.target_objects_loaded,
            "timestamp": self._timestamp,
        }


@dataclass
class SimulationState:
    """
    Manages the overall simulation state.
    
    This class serves as the root state container, coordinating
    robot state, scene state, and simulation control parameters.
    
    Attributes:
        status: Current simulation status.
        robot_state: Robot state instance.
        scene_state: Scene state instance.
        step_count: Number of simulation steps executed.
        simulation_time: Total simulation time in seconds.
        dt: Simulation time step.
        control_rate: Control loop rate in Hz.
    """
    status: SimulationStatus = SimulationStatus.IDLE
    robot_state: Optional[RobotState] = None
    scene_state: Optional[SceneState] = None
    step_count: int = 0
    simulation_time: float = 0.0
    dt: float = 0.01
    control_rate: float = 100.0
    _timestamp: float = field(default_factory=time.time)
    
    def initialize(self) -> None:
        """Initialize the simulation state."""
        self.status = SimulationStatus.INITIALIZING
        self.step_count = 0
        self.simulation_time = 0.0
        self._timestamp = time.time()
    
    def start(self) -> None:
        """Start the simulation."""
        self.status = SimulationStatus.RUNNING
        self._timestamp = time.time()
    
    def pause(self) -> None:
        """Pause the simulation."""
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.PAUSED
            self._timestamp = time.time()
    
    def resume(self) -> None:
        """Resume the simulation from pause."""
        if self.status == SimulationStatus.PAUSED:
            self.status = SimulationStatus.RUNNING
            self._timestamp = time.time()
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.status = SimulationStatus.STOPPED
        self._timestamp = time.time()
    
    def error(self) -> None:
        """Set simulation to error state."""
        self.status = SimulationStatus.ERROR
        self._timestamp = time.time()
    
    def step(self) -> None:
        """Advance simulation by one step."""
        if self.status == SimulationStatus.RUNNING:
            self.step_count += 1
            self.simulation_time += self.dt
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self.status == SimulationStatus.RUNNING
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self.status == SimulationStatus.PAUSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "step_count": self.step_count,
            "simulation_time": self.simulation_time,
            "dt": self.dt,
            "control_rate": self.control_rate,
            "robot_state": self.robot_state.to_dict() if self.robot_state else None,
            "scene_state": self.scene_state.to_dict() if self.scene_state else None,
            "timestamp": self._timestamp,
        }
