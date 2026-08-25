"""
Control Mode Implementations for Controller Layer.

This module provides control mode implementations using Simple Factory:
- IControlMode: Abstract interface for control modes
- KeyboardControlMode: Keyboard-based control mode
- ROSControlMode: ROS-based control mode
- RLControlMode: Reinforcement learning control mode
- ControlModeFactory: Factory for creating control mode instances

The Simple Factory pattern enables:
- Easy switching between control modes
- Centralized mode creation logic
- Future extensibility for new modes
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from controller.keyboard import KeyboardInputProcessor, TerminalKeyboardListener
from model.registry import Registry
from model.state import RobotState, SimulationState

logger = logging.getLogger(__name__)


class IControlMode(ABC):
    """
    Abstract interface for control modes.
    
    Defines the contract for different control mode implementations,
    enabling seamless switching between keyboard, ROS, and RL modes.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the control mode.
        
        Returns:
            True if initialization successful.
        """
        pass
    
    @abstractmethod
    def get_input(self) -> Optional[Any]:
        """
        Get input from the control source.
        
        Returns:
            Input data or None if no input available.
        """
        pass
    
    @abstractmethod
    def process_input(self, input_data: Any, robot_state: RobotState) -> np.ndarray:
        """
        Process input and compute target positions.
        
        Args:
            input_data: Raw input data.
            robot_state: Current robot state.
            
        Returns:
            Target joint positions.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release resources and cleanup."""
        pass
    
    @abstractmethod
    def get_mode_name(self) -> str:
        """
        Return the mode name identifier.
        
        Returns:
            String identifier for the mode.
        """
        pass
    
    @abstractmethod
    def get_key_bindings(self) -> Dict[str, str]:
        """
        Return key bindings or control instructions.
        
        Returns:
            Dictionary mapping keys/actions to descriptions.
        """
        pass


class KeyboardControlMode(IControlMode):
    """
    Keyboard-based control mode implementation.
    
    Provides keyboard input handling for direct robot control.
    """
    
    def __init__(
        self,
        joint_step: float = 0.1,
        gripper_step: float = 0.005,
    ):
        """
        Initialize keyboard control mode.
        
        Args:
            joint_step: Step size for joint angle changes (radians).
            gripper_step: Step size for gripper changes (meters).
        """
        self._processor = KeyboardInputProcessor(joint_step=joint_step, gripper_step=gripper_step)
        self._keyboard_handler = None
        self._running = False
        logger.info("KeyboardControlMode created")
    
    def initialize(self) -> bool:
        """Initialize keyboard handler."""
        try:
            self._keyboard_handler = TerminalKeyboardListener()
            self._keyboard_handler.start()
            self._running = True
            logger.info("Keyboard control mode initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize keyboard: {e}")
            return False
    
    def get_input(self) -> Optional[str]:
        """Get keyboard input."""
        if self._keyboard_handler:
            return self._keyboard_handler.get_key()
        return None
    
    def process_input(self, input_data: str, robot_state: RobotState) -> np.ndarray:
        """
        Process keyboard input and compute target positions.
        
        Args:
            input_data: Key string.
            robot_state: Current robot state.
            
        Returns:
            Target joint positions.
        """
        target = robot_state.target_positions.copy()
        
        if input_data is None:
            return target
        
        if input_data in ("q", "Q"):
            self._running = False
            return target
        
        target, action = self._processor.process_key(input_data, target)
        
        return robot_state.clamp_positions(target)
    
    def cleanup(self) -> None:
        """Cleanup keyboard handler."""
        if self._keyboard_handler:
            self._keyboard_handler.stop()
        self._running = False
    
    def get_mode_name(self) -> str:
        return "keyboard"
    
    def get_key_bindings(self) -> Dict[str, str]:
        return self._processor.get_key_bindings()
    
    def get_active_joint(self) -> int:
        """Return the currently active joint index."""
        return self._processor.selected_joint
    
    def is_running(self) -> bool:
        """Return whether the mode is running."""
        return self._running


class ROSControlMode(IControlMode):
    """
    ROS-based control mode implementation.
    
    Provides ROS topic integration for external robot control.
    """
    
    def __init__(
        self,
        joint_topic: str = "/real_arm/joint_angles",
        state_topic: str = "/real_arm/robot_state",
        use_real_ros: bool = False,
    ):
        """
        Initialize ROS control mode.
        
        Args:
            joint_topic: ROS topic for joint commands.
            state_topic: ROS topic for robot state.
            use_real_ros: Whether to use real ROS libraries.
        """
        self._joint_topic = joint_topic
        self._state_topic = state_topic
        self._use_real_ros = use_real_ros
        self._ros_controller = None
        self._latest_command: Optional[np.ndarray] = None
        self._command_callback: Optional[Callable] = None
        logger.info("ROSControlMode created")
    
    def initialize(self) -> bool:
        """Initialize ROS controller."""
        logger.warning("ROS controller has been removed from this project")
        return True
    
    def set_joint_config(
        self,
        joint_names: List[str],
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
    ) -> None:
        """
        Set joint configuration for ROS controller.
        
        Args:
            joint_names: List of joint names.
            joint_limits_lower: Lower joint limits.
            joint_limits_upper: Upper joint limits.
        """
        if self._ros_controller:
            self._ros_controller.joint_names = joint_names
            self._ros_controller.joint_limits_lower = joint_limits_lower
            self._ros_controller.joint_limits_upper = joint_limits_upper
    
    def _on_command(self, command) -> None:
        """Callback for ROS joint commands."""
        self._latest_command = command.to_numpy()
        if self._command_callback:
            self._command_callback(self._latest_command)
    
    def set_command_callback(self, callback: Callable) -> None:
        """
        Set callback for received commands.
        
        Args:
            callback: Function to call when command received.
        """
        self._command_callback = callback
    
    def get_input(self) -> Optional[np.ndarray]:
        """Get latest ROS command."""
        command = self._latest_command
        self._latest_command = None
        return command
    
    def process_input(self, input_data: np.ndarray, robot_state: RobotState) -> np.ndarray:
        """
        Process ROS command and compute target positions.
        
        Args:
            input_data: Joint command array.
            robot_state: Current robot state.
            
        Returns:
            Target joint positions.
        """
        if input_data is None:
            return robot_state.target_positions
        
        return robot_state.clamp_positions(input_data)
    
    def cleanup(self) -> None:
        """Cleanup ROS controller."""
        if self._ros_controller:
            self._ros_controller.stop()
    
    def get_mode_name(self) -> str:
        return "ros"
    
    def get_key_bindings(self) -> Dict[str, str]:
        return {
            "topic": self._joint_topic,
            "mode": "Subscribe to joint commands",
        }
    
    def update_robot_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        efforts: np.ndarray,
    ) -> None:
        """
        Update robot state in ROS controller.
        
        Args:
            positions: Current joint positions.
            velocities: Current joint velocities.
            efforts: Current joint efforts.
        """
        if self._ros_controller:
            self._ros_controller.update_robot_state(positions, velocities, efforts)


class RLControlMode(IControlMode):
    """
    Reinforcement learning control mode implementation.
    
    Provides interface for RL policy inference.
    """
    
    def __init__(self, policy: Optional[Any] = None):
        """
        Initialize RL control mode.
        
        Args:
            policy: RL policy for action inference.
        """
        self._policy = policy
        self._observation_builder = None
        logger.info("RLControlMode created")
    
    def initialize(self) -> bool:
        """Initialize RL control mode."""
        logger.info("RL control mode initialized")
        return True
    
    def set_policy(self, policy: Any) -> None:
        """
        Set the RL policy.
        
        Args:
            policy: Policy object with __call__ method.
        """
        self._policy = policy
    
    def set_observation_builder(self, builder: Callable) -> None:
        """
        Set the observation builder function.
        
        Args:
            builder: Function that takes robot_state and returns observation.
        """
        self._observation_builder = builder
    
    def get_input(self) -> Optional[np.ndarray]:
        """Build observation from robot state."""
        return None
    
    def process_input(self, input_data: Any, robot_state: RobotState) -> np.ndarray:
        """
        Process observation and compute action using policy.
        
        Args:
            input_data: Observation data.
            robot_state: Current robot state.
            
        Returns:
            Target joint positions (action).
        """
        if self._policy is None:
            return robot_state.target_positions
        
        try:
            if self._observation_builder:
                observation = self._observation_builder(robot_state)
            else:
                observation = np.concatenate([
                    robot_state.positions,
                    robot_state.velocities,
                    robot_state.target_positions,
                ])
            
            action = self._policy(observation)
            
            return robot_state.clamp_positions(action)
            
        except Exception as e:
            logger.error(f"Policy inference error: {e}")
            return robot_state.target_positions
    
    def cleanup(self) -> None:
        """Cleanup RL control mode."""
        pass
    
    def get_mode_name(self) -> str:
        return "rl"
    
    def get_key_bindings(self) -> Dict[str, str]:
        return {
            "mode": "RL policy inference",
            "policy": str(type(self._policy).__name__) if self._policy else "None",
        }


# Module-level registry instance for control modes
_mode_registry = Registry[IControlMode]("control_modes")
_mode_registry.register("keyboard", KeyboardControlMode)
_mode_registry.register("ros", ROSControlMode)
_mode_registry.register("rl", RLControlMode)


class ControlModeFactory:
    """Factory for creating control mode instances, backed by model.registry.Registry."""
    
    @classmethod
    def register(cls, name: str, mode_class: Type[IControlMode]) -> None:
        _mode_registry.register(name, mode_class)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> IControlMode:
        return _mode_registry.create(name, **kwargs)
    
    @classmethod
    def get_available_modes(cls) -> List[str]:
        return _mode_registry.get_available()
