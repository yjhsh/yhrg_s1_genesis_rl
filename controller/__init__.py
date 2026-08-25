"""
Controller Layer for Genesis Simulation.

This package contains the Controller components following MVC architecture:
- Control flow templates
- Input processing handlers
- Inference and execution handlers
- Unified keyboard control (controller.keyboard)
- Unified Pipeline base class (controller.pipeline)
- IK solvers (controller.ik_solver)
- Task-space controller (controller.task_space_controller)

The Controller layer is responsible for:
- Coordinating between Model and View
- Processing user inputs and external commands
- Managing control algorithms and logic
- Executing motion commands

Design Patterns:
- Template Method: BaseControlFlow for standardized control flow
- Chain of Responsibility: Input -> Inference -> Output pipeline
- Simple Factory: Control mode implementations (ROS, Keyboard, RL)
- Strategy: IK solver algorithms (DLS, Pseudoinverse)
- Facade: TaskSpaceController for IK + adaptive PD pipeline
"""

from controller.flow import (
    BaseControlFlow,
    KeyboardControlFlow,
    ROSControlFlow,
    RLControlFlow,
    ControlFlowFactory,
)
from controller.handlers import (
    IControlHandler,
    InputProcessingHandler,
    InferenceHandler,
    OutputExecutionHandler,
    ControlPipeline,
)
from controller.modes import (
    IControlMode,
    KeyboardControlMode,
    ROSControlMode,
    RLControlMode,
    ControlModeFactory,
)
from controller.keyboard import (
    KeyboardInputProcessor,
    TerminalKeyboardListener,
    KEY_BINDINGS,
)
from controller.pipeline import Pipeline
from controller.ik_solver import IIKSolver, DLSSolver, PseudoinverseSolver
from controller.task_space_controller import TaskSpaceController

__all__ = [
    "BaseControlFlow",
    "KeyboardControlFlow",
    "ROSControlFlow",
    "RLControlFlow",
    "ControlFlowFactory",
    "IControlHandler",
    "InputProcessingHandler",
    "InferenceHandler",
    "OutputExecutionHandler",
    "ControlPipeline",
    "IControlMode",
    "KeyboardControlMode",
    "ROSControlMode",
    "RLControlMode",
    "ControlModeFactory",
    "KeyboardInputProcessor",
    "TerminalKeyboardListener",
    "KEY_BINDINGS",
    "Pipeline",
    "IIKSolver",
    "DLSSolver",
    "PseudoinverseSolver",
    "TaskSpaceController",
]
