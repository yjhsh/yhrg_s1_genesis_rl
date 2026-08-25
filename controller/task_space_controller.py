"""
Task-Space Controller: delta_pose → DLS IK → Adaptive PD.

Orchestrates the complete control pipeline from task-space actions to
Genesis simulator commands:

  1. Scale raw delta_pose to physical units
  2. Compute Jacobian from current robot state
  3. Solve IK: delta_pose → delta_qpos via DLS
  4. Compute target joint positions: current_qpos + delta_qpos
  5. Map raw kp/kd from [-1,1] to physical gain ranges
  6. Set adaptive PD gains on the robot
  7. Set position target via control_dofs_position

Design Patterns:
  - Strategy: Composes an IIKSolver (swappable IK algorithm)
  - Facade: Encapsulates the multi-step IK+PD pipeline behind a single method
"""

import torch

import genesis as gs

from controller.ik_solver import IIKSolver


class TaskSpaceController:
    """Task-space controller with adaptive PD gains.

    Translates RL network outputs (delta_pose, kp, kd) into Genesis
    simulator commands using differential IK and per-step PD gain control.

    Args:
        robot: Genesis RigidEntity for the robot.
        ee_link: RigidLink for the end-effector (used for Jacobian).
        arm_dof_idx: Global DOF indices for the 6 arm joints.
        gripper_dof_idx: Global DOF indices for the 2 gripper joints.
        ik_solver: IK solver strategy (e.g. DLSSolver).
        delta_pos_scale: Scale for position delta: [-1,1] → [-scale, +scale] m.
        delta_orient_scale: Scale for orientation delta: [-1,1] → [-scale, +scale] rad.
        kp_range: (lower[6], upper[6]) gain ranges for arm joints.
        kd_range: (lower[6], upper[6]) gain ranges for arm joints.
        gripper_kp: Fixed kp for gripper joints.
        gripper_kd: Fixed kd for gripper joints.
    """

    def __init__(
        self,
        robot,
        ee_link,
        arm_dof_idx: torch.Tensor,
        gripper_dof_idx: torch.Tensor,
        ik_solver: IIKSolver,
        delta_pos_scale: float = 0.05,
        delta_orient_scale: float = 0.3,
        kp_range: tuple = None,
        kd_range: tuple = None,
        gripper_kp: float = 400.0,
        gripper_kd: float = 40.0,
    ) -> None:
        self.robot = robot
        self.ee_link = ee_link
        self.arm_dof_idx = arm_dof_idx
        self.gripper_dof_idx = gripper_dof_idx
        self.ik_solver = ik_solver

        self.delta_pos_scale = delta_pos_scale
        self.delta_orient_scale = delta_orient_scale

        # Gain ranges as tensors on the Genesis device
        device = gs.device
        self.kp_lower = torch.tensor(kp_range[0], dtype=gs.tc_float, device=device) if kp_range else None
        self.kp_upper = torch.tensor(kp_range[1], dtype=gs.tc_float, device=device) if kp_range else None
        self.kd_lower = torch.tensor(kd_range[0], dtype=gs.tc_float, device=device) if kd_range else None
        self.kd_upper = torch.tensor(kd_range[1], dtype=gs.tc_float, device=device) if kd_range else None

        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd

        # Pose scale vector [pos_scale, pos_scale, pos_scale, orient_scale, ...]
        self.pose_scale = torch.tensor(
            [delta_pos_scale] * 3 + [delta_orient_scale] * 3,
            dtype=gs.tc_float, device=device,
        )

        # Pre-allocate gripper target
        self.ARM_DOF = len(arm_dof_idx)
        self.GRIPPER_DOF = len(gripper_dof_idx)

    def compute_and_apply(
        self,
        delta_pose: torch.Tensor,
        kp_raw: torch.Tensor,
        kd_raw: torch.Tensor,
    ) -> tuple:
        """Core method: parse actions → IK → set PD gains → position control.

        Args:
            delta_pose: [n_envs, 6] raw network output for pose delta.
            kp_raw: [n_envs, 6] raw network output for proportional gains ∈ [-1,1].
            kd_raw: [n_envs, 6] raw network output for derivative gains ∈ [-1,1].

        Returns:
            target_qpos: [n_envs, 6] target arm joint positions.
            kp_applied: [n_envs, 6] applied proportional gains.
            kd_applied: [n_envs, 6] applied derivative gains.
        """
        n_envs = delta_pose.shape[0]
        device = delta_pose.device

        # 1. Scale delta_pose to physical units
        error = delta_pose * self.pose_scale  # [n_envs, 6]

        # 2. Get Jacobian (arm columns only: first 6 DOFs of the entity)
        jac = self.robot.get_jacobian(self.ee_link)[:, :, :self.ARM_DOF]  # [n_envs, 6, 6]

        # 3. DLS IK: task-space error → joint-space delta
        dq = self.ik_solver.solve(jac, error)  # [n_envs, 6]

        # 4. Target joint positions = current + delta
        current_qpos = self.robot.get_qpos()[:, :self.ARM_DOF]  # [n_envs, 6]
        target_qpos = current_qpos + dq

        # 5. Map kp/kd from [-1, 1] to physical ranges
        kp_applied = self.kp_lower + (kp_raw + 1.0) / 2.0 * (self.kp_upper - self.kp_lower)
        kd_applied = self.kd_lower + (kd_raw + 1.0) / 2.0 * (self.kd_upper - self.kd_lower)

        # 6. Set adaptive PD gains for arm joints
        self.robot.set_dofs_kp(kp_applied, self.arm_dof_idx)
        self.robot.set_dofs_kv(kd_applied, self.arm_dof_idx)

        # Set fixed PD gains for gripper joints
        gripper_kp = torch.full((n_envs, self.GRIPPER_DOF), self.gripper_kp, dtype=gs.tc_float, device=device)
        gripper_kd = torch.full((n_envs, self.GRIPPER_DOF), self.gripper_kd, dtype=gs.tc_float, device=device)
        self.robot.set_dofs_kp(gripper_kp, self.gripper_dof_idx)
        self.robot.set_dofs_kv(gripper_kd, self.gripper_dof_idx)

        # 7. Position control: arm target + gripper closed
        gripper_target = torch.zeros((n_envs, self.GRIPPER_DOF), dtype=gs.tc_float, device=device)
        full_target = torch.cat([target_qpos, gripper_target], dim=-1)
        self.robot.control_dofs_position(full_target)

        return target_qpos, kp_applied, kd_applied

    def set_initial_gains(self, n_envs: int) -> None:
        """Set PD gains to mid-range values (for initialization / reset).

        Args:
            n_envs: Number of parallel environments.
        """
        device = gs.device
        kp_mid = (self.kp_lower + self.kp_upper) / 2.0
        kd_mid = (self.kd_lower + self.kd_upper) / 2.0

        kp_mid_expanded = kp_mid.unsqueeze(0).expand(n_envs, -1)
        kd_mid_expanded = kd_mid.unsqueeze(0).expand(n_envs, -1)

        self.robot.set_dofs_kp(kp_mid_expanded, self.arm_dof_idx)
        self.robot.set_dofs_kv(kd_mid_expanded, self.arm_dof_idx)

        gripper_kp = torch.full((n_envs, self.GRIPPER_DOF), self.gripper_kp, dtype=gs.tc_float, device=device)
        gripper_kd = torch.full((n_envs, self.GRIPPER_DOF), self.gripper_kd, dtype=gs.tc_float, device=device)
        self.robot.set_dofs_kp(gripper_kp, self.gripper_dof_idx)
        self.robot.set_dofs_kv(gripper_kd, self.gripper_dof_idx)
