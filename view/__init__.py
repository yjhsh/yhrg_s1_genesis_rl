"""
View Layer for Genesis Simulation.

This package contains the View components following MVC architecture:
- Simulator proxy interfaces
- Simulation flow handlers
- Visualization and display components

The View layer is responsible for:
- Communicating with the simulator backend
- Managing simulation flow (init, run, pause, reset)
- Displaying simulation state
- User interface components

Design Patterns:
- Proxy Pattern: SimulatorProxy for decoupled communication
- Chain of Responsibility: Simulation flow control handlers
"""

from view.proxy import (
    ISimulatorProxy,
    GenesisSimulatorProxy,
    SimulatorProxyFactory,
)
from view.flow import (
    ISimulationFlowHandler,
    InitFlowHandler,
    RunFlowHandler,
    PauseFlowHandler,
    ResetFlowHandler,
    SimulationFlowPipeline,
)

__all__ = [
    "ISimulatorProxy",
    "GenesisSimulatorProxy",
    "SimulatorProxyFactory",
    "ISimulationFlowHandler",
    "InitFlowHandler",
    "RunFlowHandler",
    "PauseFlowHandler",
    "ResetFlowHandler",
    "SimulationFlowPipeline",
]
