"""
Simulation Flow Handlers for View Layer.

This module provides simulation flow control using Chain of Responsibility:
- ISimulationFlowHandler: Abstract handler interface
- InitFlowHandler: Handles simulation initialization
- RunFlowHandler: Handles simulation running
- PauseFlowHandler: Handles simulation pause
- ResetFlowHandler: Handles simulation reset
- SimulationFlowPipeline: Orchestrates the handler chain

The Chain of Responsibility pattern enables:
- Flexible flow control composition
- Easy addition of new flow handlers
- Decoupled flow management
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from model.state import SimulationState, SimulationStatus, RobotState
from controller.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ISimulationFlowHandler(ABC):
    """
    Abstract handler interface for simulation flow control.
    
    Each handler processes a specific flow step and passes control
    to the next handler in the chain.
    """
    
    def __init__(self):
        self._next_handler: Optional["ISimulationFlowHandler"] = None
    
    def set_next(self, handler: "ISimulationFlowHandler") -> "ISimulationFlowHandler":
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
        Process the flow step and optionally pass to next handler.
        
        Args:
            context: Flow context containing state and parameters.
            
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


class InitFlowHandler(ISimulationFlowHandler):
    """
    Handler for simulation initialization.
    
    Responsible for:
    - Initializing simulator
    - Creating scene
    - Loading robot and objects
    - Configuring physics parameters
    """
    
    def __init__(
        self,
        simulator_proxy,
        scene_config,
        robot_config,
        show_viewer: bool = True,
    ):
        """
        Initialize the init flow handler.
        
        Args:
            simulator_proxy: Simulator proxy instance.
            scene_config: Scene configuration.
            robot_config: Robot configuration.
            show_viewer: Whether to show the viewer.
        """
        super().__init__()
        self._simulator_proxy = simulator_proxy
        self._scene_config = scene_config
        self._robot_config = robot_config
        self._show_viewer = show_viewer
        logger.info("InitFlowHandler created")
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the simulation.
        
        Args:
            context: Context containing initialization parameters.
            
        Returns:
            Updated context with initialized entities.
        """
        logger.info("Initializing simulation...")
        
        sim_state = context.get("simulation_state")
        if sim_state:
            sim_state.initialize()
        
        if not self._simulator_proxy.is_initialized():
            backend = context.get("backend", "gpu")
            if not self._simulator_proxy.initialize(backend):
                logger.error("Failed to initialize simulator")
                if sim_state:
                    sim_state.error()
                return self._pass_to_next(context)
        
        camera_config = self._scene_config.camera
        scene = self._simulator_proxy.create_scene(
            camera_pos=camera_config.position,
            camera_lookat=camera_config.lookat,
            camera_fov=camera_config.fov,
            show_viewer=self._show_viewer,
            dt=context.get("dt", 0.01),
        )
        
        plane = self._simulator_proxy.add_plane(scene)
        
        context["scene"] = scene
        context["plane"] = plane
        context["gs"] = self._simulator_proxy.get_simulator_module()
        
        robot = self._simulator_proxy.load_urdf(
            scene,
            self._robot_config["urdf_path"],
            pos=(0.0, 0.0, 0.0),
            fixed=self._robot_config.get("fixed_base", True),
        )
        
        joint_names = self._robot_config.get("joint_names", ())
        motor_dof_idx = []
        for name in joint_names:
            try:
                joint = self._simulator_proxy.get_joint(robot, name)
                motor_dof_idx.append(joint.dofs_idx_local[0])
            except Exception as e:
                logger.warning(f"Could not find joint '{name}': {e}")
        
        context["robot"] = robot
        context["motor_dof_idx"] = motor_dof_idx
        context["robot_config"] = self._robot_config
        
        control_gains = self._robot_config.get("control_gains", {})
        force_limits = self._robot_config.get("force_limits", {})
        
        if "kp" in control_gains:
            self._simulator_proxy.set_joint_kp(
                robot, np.array(control_gains["kp"]), motor_dof_idx
            )
        
        if "kv" in control_gains:
            self._simulator_proxy.set_joint_kv(
                robot, np.array(control_gains["kv"]), motor_dof_idx
            )
        
        if "lower" in force_limits and "upper" in force_limits:
            self._simulator_proxy.set_joint_force_range(
                robot,
                np.array(force_limits["lower"]),
                np.array(force_limits["upper"]),
                motor_dof_idx,
            )
        
        self._simulator_proxy.build_scene(scene)
        
        default_positions = self._robot_config.get("default_positions", {})
        initial_pos = default_positions.get("neutral", np.zeros(len(motor_dof_idx)))
        
        for _ in range(50):
            self._simulator_proxy.set_joint_positions(robot, initial_pos, motor_dof_idx)
            self._simulator_proxy.step(scene)
        
        if sim_state:
            sim_state.start()
        
        context["initialized"] = True
        logger.info("Simulation initialized successfully")
        
        return self._pass_to_next(context)


class RunFlowHandler(ISimulationFlowHandler):
    """
    Handler for simulation running.
    
    Responsible for:
    - Executing simulation steps
    - Updating robot state
    - Processing motion control
    """
    
    def __init__(self, simulator_proxy):
        """
        Initialize the run flow handler.
        
        Args:
            simulator_proxy: Simulator proxy instance.
        """
        super().__init__()
        self._simulator_proxy = simulator_proxy
        self._target_position: Optional[np.ndarray] = None
        logger.info("RunFlowHandler created")
    
    def set_target_position(self, position: np.ndarray) -> None:
        """
        Set target joint positions for position control.
        
        Args:
            position: Target position array.
        """
        self._target_position = position
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Args:
            context: Context containing scene and robot.
            
        Returns:
            Updated context after step.
        """
        scene = context.get("scene")
        robot = context.get("robot")
        motor_dof_idx = context.get("motor_dof_idx", [])
        sim_state = context.get("simulation_state")
        
        if scene is None or robot is None:
            logger.warning("Scene or robot not available")
            return self._pass_to_next(context)
        
        if sim_state and not sim_state.is_running():
            return self._pass_to_next(context)
        
        target = context.get("target_position", self._target_position)
        if target is not None:
            self._simulator_proxy.control_joint_positions(robot, target, motor_dof_idx)
        
        self._simulator_proxy.step(scene)
        
        if sim_state:
            sim_state.step()
        
        positions = self._simulator_proxy.get_joint_positions(robot, motor_dof_idx)
        velocities = self._simulator_proxy.get_joint_velocities(robot, motor_dof_idx)
        efforts = self._simulator_proxy.get_joint_efforts(robot, motor_dof_idx)
        control_force = self._simulator_proxy.get_control_force(robot, motor_dof_idx)
        
        context["current_positions"] = positions
        context["current_velocities"] = velocities
        context["current_efforts"] = efforts
        context["control_force"] = control_force
        
        robot_state = context.get("robot_state")
        if robot_state:
            robot_state.update_positions(positions)
            robot_state.update_velocities(velocities)
            robot_state.update_efforts(efforts)
        
        return self._pass_to_next(context)


class PauseFlowHandler(ISimulationFlowHandler):
    """
    Handler for simulation pause.
    
    Responsible for:
    - Pausing simulation execution
    - Maintaining current state
    """
    
    def __init__(self):
        """Initialize the pause flow handler."""
        super().__init__()
        logger.info("PauseFlowHandler created")
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pause the simulation.
        
        Args:
            context: Context containing simulation state.
            
        Returns:
            Updated context after pause.
        """
        sim_state = context.get("simulation_state")
        if sim_state:
            sim_state.pause()
            logger.info("Simulation paused")
        
        return self._pass_to_next(context)


class ResetFlowHandler(ISimulationFlowHandler):
    """
    Handler for simulation reset.
    
    Responsible for:
    - Resetting robot to initial position
    - Resetting simulation state
    - Clearing any error conditions
    """
    
    def __init__(self, simulator_proxy):
        """
        Initialize the reset flow handler.
        
        Args:
            simulator_proxy: Simulator proxy instance.
        """
        super().__init__()
        self._simulator_proxy = simulator_proxy
        logger.info("ResetFlowHandler created")
    
    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset the simulation.
        
        Args:
            context: Context containing scene and robot.
            
        Returns:
            Updated context after reset.
        """
        scene = context.get("scene")
        robot = context.get("robot")
        motor_dof_idx = context.get("motor_dof_idx", [])
        robot_config = context.get("robot_config", {})
        sim_state = context.get("simulation_state")
        
        if scene is None or robot is None:
            logger.warning("Scene or robot not available for reset")
            return self._pass_to_next(context)
        
        default_positions = robot_config.get("default_positions", {})
        initial_pos = default_positions.get("neutral", np.zeros(len(motor_dof_idx)))
        
        for _ in range(50):
            self._simulator_proxy.set_joint_positions(robot, initial_pos, motor_dof_idx)
            self._simulator_proxy.step(scene)
        
        if sim_state:
            sim_state.step_count = 0
            sim_state.simulation_time = 0.0
            sim_state.start()
        
        context["target_position"] = initial_pos
        logger.info("Simulation reset to initial position")
        
        return self._pass_to_next(context)


class SimulationFlowPipeline(Pipeline[ISimulationFlowHandler]):
    """
    Orchestrates the simulation flow handler chain.
    
    Inherits from controller.pipeline.Pipeline for common Pipeline logic.
    Adds simulation-specific convenience methods (pause, reset, set_target_position).
    """
    
    def __init__(self):
        super().__init__(name="SimulationFlowPipeline")
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a single simulation step.
        
        Finds the RunFlowHandler and executes it.
        
        Returns:
            Updated context after step.
        """
        run_handler = None
        for handler in self._handlers:
            if isinstance(handler, RunFlowHandler):
                run_handler = handler
                break
        
        if run_handler is None:
            raise RuntimeError("No RunFlowHandler in pipeline")
        
        return run_handler.handle(self._context)
    
    def pause(self) -> None:
        """Pause the simulation."""
        for handler in self._handlers:
            if isinstance(handler, PauseFlowHandler):
                handler.handle(self._context)
                break
    
    def reset(self) -> None:
        """Reset the simulation."""
        for handler in self._handlers:
            if isinstance(handler, ResetFlowHandler):
                handler.handle(self._context)
                break
    
    def set_target_position(self, position: np.ndarray) -> None:
        """
        Set target position for motion control.
        
        Args:
            position: Target joint positions.
        """
        for handler in self._handlers:
            if isinstance(handler, RunFlowHandler):
                handler.set_target_position(position)
                break
        self._context["target_position"] = position
