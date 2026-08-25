"""
Control Handlers for Controller Layer.

This module provides control pipeline handlers using Chain of Responsibility:
- IControlHandler: Abstract handler interface
- InputProcessingHandler: Processes input from various sources
- InferenceHandler: Computes control actions
- OutputExecutionHandler: Executes control commands
- ControlPipeline: Orchestrates the handler chain

The Chain of Responsibility pattern enables:
- Flexible control pipeline composition
- Easy addition of new processing stages
- Decoupled control logic
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from model.state import RobotState
from controller.keyboard import KeyboardInputProcessor
from controller.pipeline import Pipeline

logger = logging.getLogger(__name__)


class IControlHandler(ABC):
    """
    Abstract handler interface for control pipeline.
    
    Each handler processes a specific stage of the control pipeline
    and passes control to the next handler in the chain.
    """
    
    def __init__(self):
        self._next_handler: Optional["IControlHandler"] = None
    
    def set_next(self, handler: "IControlHandler") -> "IControlHandler":
        """
        Set the next handler in the chain.
        
        Args:
            handler: The next handler to execute.
            
        Returns:
            The handler that was set, for method chaining.
        """
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the control stage and optionally pass to next handler.
        
        Args:
            context: Control context containing state and parameters.
            
        Returns:
            Updated context after processing.
        """
        pass
    
    def _pass_to_next(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pass the context to the next handler if one exists.
        
        Args:
            context: The context to pass forward.
            
        Returns:
            Context after next handler processing.
        """
        if self._next_handler is not None:
            return self._next_handler.handle(context)
        return context


class InputProcessingHandler(IControlHandler):
    """
    Handler for input processing stage.
    
    Responsible for:
    - Processing keyboard, ROS, or other input sources
    - Validating input data
    - Converting input to standardized format
    """
    
    def __init__(self, input_source: Optional[Any] = None):
        """
        Initialize the input processing handler.
        
        Args:
            input_source: Optional input source (keyboard, ROS, etc.).
        """
        super().__init__()
        self._input_source = input_source
        self._last_input: Optional[Any] = None
        logger.info("InputProcessingHandler created")
    
    def set_input_source(self, source: Any) -> None:
        """
        Set the input source.
        
        Args:
            source: Input source object.
        """
        self._input_source = source
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input from the configured source.
        
        Args:
            context: Control context.
            
        Returns:
            Updated context with processed input.
        """
        input_data = None
        
        if self._input_source is not None:
            try:
                if hasattr(self._input_source, 'get_input'):
                    input_data = self._input_source.get_input()
                elif hasattr(self._input_source, 'get_key'):
                    input_data = self._input_source.get_key()
                elif callable(self._input_source):
                    input_data = self._input_source()
            except Exception as e:
                logger.error(f"Error getting input: {e}")
        
        if input_data is not None:
            self._last_input = input_data
            context["input_data"] = input_data
            context["input_processed"] = True
        else:
            context["input_processed"] = False
        
        return self._pass_to_next(context)
    
    def get_last_input(self) -> Any:
        """Return the last processed input."""
        return self._last_input


class InferenceHandler(IControlHandler):
    """
    Handler for inference/computation stage.
    
    Responsible for:
    - Computing control actions from input
    - Applying control algorithms
    - Generating target positions/velocities
    """
    
    def __init__(
        self,
        robot_state: Optional[RobotState] = None,
        inference_func: Optional[callable] = None,
    ):
        """
        Initialize the inference handler.
        
        Args:
            robot_state: Robot state reference.
            inference_func: Optional custom inference function.
        """
        super().__init__()
        self._robot_state = robot_state
        self._inference_func = inference_func
        self._processor = KeyboardInputProcessor()
        logger.info("InferenceHandler created")
    
    def set_robot_state(self, robot_state: RobotState) -> None:
        """
        Set the robot state reference.
        
        Args:
            robot_state: Robot state instance.
        """
        self._robot_state = robot_state
    
    def set_inference_function(self, func: callable) -> None:
        """
        Set a custom inference function.
        
        Args:
            func: Function that takes (input_data, context) and returns action.
        """
        self._inference_func = func
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute control action from input.
        
        Args:
            context: Control context with input data.
            
        Returns:
            Updated context with computed action.
        """
        input_data = context.get("input_data")
        action = None
        
        if self._inference_func is not None:
            try:
                action = self._inference_func(input_data, context)
            except Exception as e:
                logger.error(f"Inference function error: {e}")
        
        elif input_data is not None:
            action = self._default_inference(input_data, context)
        
        if action is not None:
            context["action"] = action
            context["action_computed"] = True
            
            if self._robot_state:
                self._robot_state.set_target_positions(action)
        else:
            context["action_computed"] = False
        
        return self._pass_to_next(context)
    
    def _default_inference(self, input_data: Any, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Default inference logic for keyboard input.

        Args:
            input_data: Input data (key string for keyboard).
            context: Control context.

        Returns:
            Computed action or None.
        """
        if self._robot_state is None:
            return None

        target = self._robot_state.target_positions.copy()

        if isinstance(input_data, str):
            target, action = self._processor.process_key(input_data, target)
            if action == "quit":
                return None
            return self._robot_state.clamp_positions(target)

        elif isinstance(input_data, np.ndarray):
            return self._robot_state.clamp_positions(input_data)

        return None
    
    def set_active_joint(self, joint_idx: int) -> None:
        self._processor._selected_joint = joint_idx
    
    def set_step_sizes(self, joint_step: float, gripper_step: float) -> None:
        self._processor = KeyboardInputProcessor(joint_step=joint_step, gripper_step=gripper_step)


class OutputExecutionHandler(IControlHandler):
    """
    Handler for output execution stage.
    
    Responsible for:
    - Executing control commands on the robot
    - Communicating with the simulator
    - Handling execution errors
    """
    
    def __init__(
        self,
        simulator_proxy=None,
        robot=None,
        motor_dof_idx: Optional[List[int]] = None,
    ):
        """
        Initialize the output execution handler.
        
        Args:
            simulator_proxy: Simulator proxy instance.
            robot: Robot entity.
            motor_dof_idx: Motor DOF indices.
        """
        super().__init__()
        self._simulator_proxy = simulator_proxy
        self._robot = robot
        self._motor_dof_idx = motor_dof_idx or []
        logger.info("OutputExecutionHandler created")
    
    def set_simulator_proxy(self, proxy) -> None:
        """
        Set the simulator proxy.
        
        Args:
            proxy: Simulator proxy instance.
        """
        self._simulator_proxy = proxy
    
    def set_robot(self, robot: Any, motor_dof_idx: List[int]) -> None:
        """
        Set the robot entity and motor indices.
        
        Args:
            robot: Robot entity.
            motor_dof_idx: Motor DOF indices.
        """
        self._robot = robot
        self._motor_dof_idx = motor_dof_idx
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the control action.
        
        Args:
            context: Control context with action to execute.
            
        Returns:
            Updated context after execution.
        """
        action = context.get("action")
        
        if action is None:
            context["execution_success"] = False
            return self._pass_to_next(context)
        
        if self._simulator_proxy is None or self._robot is None:
            logger.warning("Simulator proxy or robot not set")
            context["execution_success"] = False
            return self._pass_to_next(context)
        
        try:
            control_mode = context.get("control_mode", "position")
            
            if control_mode == "position":
                self._simulator_proxy.control_joint_positions(
                    self._robot, action, self._motor_dof_idx
                )
            elif control_mode == "velocity":
                self._simulator_proxy.control_joint_velocities(
                    self._robot, action, self._motor_dof_idx
                )
            elif control_mode == "effort":
                self._simulator_proxy.control_joint_efforts(
                    self._robot, action, self._motor_dof_idx
                )
            
            context["execution_success"] = True
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            context["execution_success"] = False
            context["execution_error"] = str(e)
        
        return self._pass_to_next(context)


class ControlPipeline(Pipeline[IControlHandler]):
    """
    Orchestrates the control handler chain.

    Inherits from controller.pipeline.Pipeline for common Pipeline logic.
    The pipeline follows the standard control flow:
    Input -> Inference -> Output
    """

    def __init__(self):
        super().__init__(name="ControlPipeline")

    def reset(self) -> None:
        """Reset the pipeline context."""
        self.clear_context()
