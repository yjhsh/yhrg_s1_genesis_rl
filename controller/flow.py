"""
Control Flow Templates for Controller Layer.

This module provides control flow implementations using Template Method Pattern:
- BaseControlFlow: Abstract base class defining control flow template
- KeyboardControlFlow: Keyboard-based control implementation
- ROSControlFlow: ROS-based control implementation
- RLControlFlow: Reinforcement learning control implementation
- ControlFlowFactory: Factory for creating control flow instances

The Template Method pattern enables:
- Standardized control flow structure
- Customizable steps for different control modes
- Code reuse across control implementations
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from model.state import RobotState, SimulationState
from controller.keyboard import KeyboardInputProcessor, TerminalKeyboardListener

logger = logging.getLogger(__name__)


class BaseControlFlow(ABC):
    """
    Abstract base class defining the control flow template.
    
    This class implements the Template Method pattern, defining the
    skeleton of the control algorithm while allowing subclasses to
    override specific steps.
    
    Template Method Steps:
    1. initialize(): Setup control flow resources
    2. process_input(): Process input from control source
    3. compute_action(): Compute control action (inference)
    4. execute_action(): Execute the computed action
    5. update_state(): Update internal state
    6. cleanup(): Release resources
    
    Attributes:
        _robot_state: Reference to robot state.
        _simulation_state: Reference to simulation state.
        _running: Flag indicating if control loop is active.
    """
    
    def __init__(
        self,
        robot_state: Optional[RobotState] = None,
        simulation_state: Optional[SimulationState] = None,
    ):
        """
        Initialize the control flow.
        
        Args:
            robot_state: Robot state reference.
            simulation_state: Simulation state reference.
        """
        self._robot_state = robot_state
        self._simulation_state = simulation_state
        self._running = False
        self._step_count = 0
        self._last_update_time = 0.0
        logger.info(f"{self.__class__.__name__} created")
    
    def run(self) -> None:
        """
        Execute the control flow using template method.
        
        This method defines the skeleton of the control algorithm.
        Subclasses implement specific steps by overriding hook methods.
        """
        if not self.initialize():
            logger.error("Failed to initialize control flow")
            return
        
        self._running = True
        self._step_count = 0
        logger.info("Control flow started")
        
        try:
            while self._running:
                start_time = time.time()
                
                input_data = self.process_input()
                if input_data is None:
                    time.sleep(0.001)
                    continue
                
                action = self.compute_action(input_data)
                
                self.execute_action(action)
                
                self.update_state()
                
                self._step_count += 1
                
                elapsed = time.time() - start_time
                if elapsed < 0.001:
                    time.sleep(0.001 - elapsed)
                    
        except KeyboardInterrupt:
            logger.info("Control flow interrupted")
        finally:
            self.cleanup()
            logger.info(f"Control flow ended. Total steps: {self._step_count}")
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a single control step.
        
        Returns:
            Dictionary containing step results.
        """
        input_data = self.process_input()
        action = self.compute_action(input_data) if input_data is not None else None
        self.execute_action(action)
        self.update_state()
        self._step_count += 1
        
        return {
            "step": self._step_count,
            "input": input_data,
            "action": action,
        }
    
    def stop(self) -> None:
        """Stop the control flow."""
        self._running = False
        logger.info("Control flow stop requested")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize control flow resources.
        
        Returns:
            True if initialization successful.
        """
        pass
    
    @abstractmethod
    def process_input(self) -> Optional[Any]:
        """
        Process input from control source.
        
        Returns:
            Processed input data or None if no input.
        """
        pass
    
    @abstractmethod
    def compute_action(self, input_data: Any) -> np.ndarray:
        """
        Compute control action from input.
        
        Args:
            input_data: Processed input data.
            
        Returns:
            Computed action (joint positions/velocities/efforts).
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: Optional[np.ndarray]) -> None:
        """
        Execute the computed action.
        
        Args:
            action: Computed action to execute.
        """
        pass
    
    def update_state(self) -> None:
        """Update internal state after action execution."""
        self._last_update_time = time.time()
    
    def cleanup(self) -> None:
        """Release resources and cleanup."""
        pass
    
    @property
    def step_count(self) -> int:
        """Return the current step count."""
        return self._step_count
    
    @property
    def is_running(self) -> bool:
        """Return whether the control flow is running."""
        return self._running


class KeyboardControlFlow(BaseControlFlow):
    """
    Keyboard-based control flow implementation.
    
    Implements control flow for keyboard-based robot control,
    translating keyboard input into joint commands.
    """
    
    def __init__(
        self,
        robot_state: RobotState,
        simulation_state: SimulationState,
        joint_step: float = 0.1,
        gripper_step: float = 0.005,
    ):
        """
        Initialize keyboard control flow.
        
        Args:
            robot_state: Robot state reference.
            simulation_state: Simulation state reference.
            joint_step: Step size for joint angle changes.
            gripper_step: Step size for gripper changes.
        """
        super().__init__(robot_state, simulation_state)
        self._processor = KeyboardInputProcessor(joint_step=joint_step, gripper_step=gripper_step)
        self._keyboard_handler = None
        self._target_position: Optional[np.ndarray] = None
    
    def initialize(self) -> bool:
        """Initialize keyboard handler."""
        try:
            self._keyboard_handler = TerminalKeyboardListener()
            self._keyboard_handler.start()
            
            if self._robot_state:
                self._target_position = np.zeros(self._robot_state.num_joints)
            
            logger.info("Keyboard control initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize keyboard: {e}")
            return False
    
    def process_input(self) -> Optional[str]:
        """Process keyboard input."""
        if self._keyboard_handler:
            return self._keyboard_handler.get_key()
        return None
    
    def compute_action(self, input_data: str) -> np.ndarray:
        """Compute action from keyboard input."""
        if self._target_position is None:
            return np.zeros(8)
        
        if input_data in ("q", "Q"):
            self.stop()
            return self._target_position
        
        self._target_position, action = self._processor.process_key(input_data, self._target_position)
        
        if self._robot_state:
            self._target_position = self._robot_state.clamp_positions(self._target_position)
        
        return self._target_position
    
    def execute_action(self, action: Optional[np.ndarray]) -> None:
        """Execute action by setting target position."""
        if action is not None and self._robot_state:
            self._robot_state.set_target_positions(action)
    
    def cleanup(self) -> None:
        """Cleanup keyboard handler."""
        if self._keyboard_handler:
            self._keyboard_handler.stop()
    
    def get_key_bindings(self) -> Dict[str, str]:
        """Return key bindings for display."""
        return self._processor.get_key_bindings()


class ROSControlFlow(BaseControlFlow):
    """
    ROS-based control flow implementation.
    
    Implements control flow for ROS-based robot control,
    receiving commands from ROS topics.
    """
    
    def __init__(
        self,
        robot_state: RobotState,
        simulation_state: SimulationState,
        joint_topic: str = "/real_arm/joint_angles",
        use_real_ros: bool = False,
    ):
        """
        Initialize ROS control flow.
        
        Args:
            robot_state: Robot state reference.
            simulation_state: Simulation state reference.
            joint_topic: ROS topic for joint commands.
            use_real_ros: Whether to use real ROS libraries.
        """
        super().__init__(robot_state, simulation_state)
        self._joint_topic = joint_topic
        self._use_real_ros = use_real_ros
        self._ros_controller = None
        self._latest_command: Optional[np.ndarray] = None
    
    def initialize(self) -> bool:
        """Initialize ROS controller."""
        logger.warning("ROS controller has been removed from this project")
        return True
    
    def _on_command(self, command) -> None:
        """Callback for ROS joint commands."""
        self._latest_command = command.to_numpy()
    
    def process_input(self) -> Optional[np.ndarray]:
        """Process ROS input."""
        command = self._latest_command
        self._latest_command = None
        return command
    
    def compute_action(self, input_data: np.ndarray) -> np.ndarray:
        """Compute action from ROS command."""
        if input_data is None:
            if self._robot_state:
                return self._robot_state.target_positions
            return np.zeros(8)
        
        if self._robot_state:
            return self._robot_state.clamp_positions(input_data)
        return input_data
    
    def execute_action(self, action: Optional[np.ndarray]) -> None:
        """Execute action by setting target position."""
        if action is not None and self._robot_state:
            self._robot_state.set_target_positions(action)
    
    def cleanup(self) -> None:
        """Cleanup ROS controller."""
        if self._ros_controller:
            self._ros_controller.stop()


class RLControlFlow(BaseControlFlow):
    """
    Reinforcement learning control flow implementation.
    
    Implements control flow for RL-based robot control,
    providing interface for policy inference.
    """
    
    def __init__(
        self,
        robot_state: RobotState,
        simulation_state: SimulationState,
        policy: Optional[Any] = None,
    ):
        """
        Initialize RL control flow.
        
        Args:
            robot_state: Robot state reference.
            simulation_state: Simulation state reference.
            policy: RL policy for action inference.
        """
        super().__init__(robot_state, simulation_state)
        self._policy = policy
        self._observation: Optional[np.ndarray] = None
    
    def initialize(self) -> bool:
        """Initialize RL control."""
        logger.info("RL control initialized")
        return True
    
    def process_input(self) -> Optional[np.ndarray]:
        """Process observation from environment."""
        if self._robot_state:
            self._observation = np.concatenate([
                self._robot_state.positions,
                self._robot_state.velocities,
                self._robot_state.target_positions,
            ])
        return self._observation
    
    def compute_action(self, input_data: np.ndarray) -> np.ndarray:
        """Compute action using RL policy."""
        if self._policy is None:
            if self._robot_state:
                return self._robot_state.target_positions
            return np.zeros(8)
        
        try:
            action = self._policy(input_data)
            return action
        except Exception as e:
            logger.error(f"Policy inference failed: {e}")
            return np.zeros(8)
    
    def execute_action(self, action: Optional[np.ndarray]) -> None:
        """Execute action by setting target position."""
        if action is not None and self._robot_state:
            self._robot_state.set_target_positions(action)
    
    def set_policy(self, policy: Any) -> None:
        """
        Set the RL policy.
        
        Args:
            policy: Policy object with __call__ method.
        """
        self._policy = policy


class ControlFlowFactory:
    """
    Factory for creating control flow instances.
    
    Implements the Simple Factory pattern to create appropriate
    control flow instances based on the control mode.
    """
    
    @staticmethod
    def create(
        mode: str,
        robot_state: RobotState,
        simulation_state: SimulationState,
        **kwargs,
    ) -> BaseControlFlow:
        """
        Create a control flow instance.
        
        Args:
            mode: Control mode ("keyboard", "ros", "rl").
            robot_state: Robot state reference.
            simulation_state: Simulation state reference.
            **kwargs: Additional arguments for specific modes.
            
        Returns:
            BaseControlFlow instance.
            
        Raises:
            ValueError: If mode is unknown.
        """
        if mode == "keyboard":
            return KeyboardControlFlow(
                robot_state=robot_state,
                simulation_state=simulation_state,
                joint_step=kwargs.get("joint_step", 0.1),
                gripper_step=kwargs.get("gripper_step", 0.005),
            )
        elif mode == "ros":
            return ROSControlFlow(
                robot_state=robot_state,
                simulation_state=simulation_state,
                joint_topic=kwargs.get("joint_topic", "/real_arm/joint_angles"),
                use_real_ros=kwargs.get("use_real_ros", False),
            )
        elif mode == "rl":
            return RLControlFlow(
                robot_state=robot_state,
                simulation_state=simulation_state,
                policy=kwargs.get("policy", None),
            )
        else:
            raise ValueError(f"Unknown control mode: {mode}")
