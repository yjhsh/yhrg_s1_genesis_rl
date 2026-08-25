"""
IK Solver implementations for task-space control.

Provides inverse kinematics solvers that map task-space errors (position +
orientation) to joint-space deltas. Uses the Strategy pattern so different
IK algorithms can be swapped at runtime.

Solvers:
  - DLSSolver: Damped Least-Squares (Levenberg-Marquardt style)
  - PseudoinverseSolver: Moore-Penrose pseudoinverse

Usage:
    solver = DLSSolver(damping=0.01)
    dq = solver.solve(jacobian, error)  # error: [n_envs, 6], dq: [n_envs, n_arm_dofs]
"""

from abc import ABC, abstractmethod

import torch


class IIKSolver(ABC):
    """Abstract interface for inverse kinematics solvers (Strategy Pattern).

    All solvers map a task-space error vector to a joint-space delta.
    The error vector convention is [position_error(3), orientation_error(3)],
    where orientation error is in axis-angle (rotation vector) representation.
    """

    @abstractmethod
    def solve(self, jacobian: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """Compute joint-space delta from task-space error.

        Args:
            jacobian: Spatial Jacobian [n_envs, 6, n_arm_dofs].
            error: Task-space error [n_envs, 6] (pos 3 + orient 3).

        Returns:
            dq: Joint-space delta [n_envs, n_arm_dofs].
        """
        ...


class DLSSolver(IIKSolver):
    """Damped Least-Squares (DLS) IK solver.

    Also known as Levenberg-Marquardt method. Adds a damping term to the
    Jacobian transpose product to handle singularities and ill-conditioning:

        dq = J^T @ (J @ J^T + λ²I)^{-1} @ error

    Args:
        damping: Damping coefficient λ. Higher values produce more
            conservative (smaller) joint deltas near singularities.
    """

    def __init__(self, damping: float = 0.01) -> None:
        self.damping = damping

    def solve(self, jacobian: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        JJT = jacobian @ jacobian.transpose(-1, -2)
        n = JJT.shape[-1]
        damped = JJT + self.damping ** 2 * torch.eye(n, device=jacobian.device, dtype=jacobian.dtype)
        dq = (jacobian.transpose(-1, -2) @ torch.linalg.solve(damped, error.unsqueeze(-1))).squeeze(-1)
        return dq


class PseudoinverseSolver(IIKSolver):
    """Moore-Penrose pseudoinverse IK solver.

    Uses the pseudoinverse of the Jacobian to map task-space error to
    joint-space delta:

        dq = J^+ @ error

    Simpler than DLS but provides no singularity protection.
    """

    def solve(self, jacobian: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        dq = (torch.linalg.pinv(jacobian) @ error.unsqueeze(-1)).squeeze(-1)
        return dq
