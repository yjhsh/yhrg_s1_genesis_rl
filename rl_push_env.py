"""
Push Environment for YHRG S1 Robotic Arm using Genesis Simulator.

Non-prehensile manipulation task: push a cube on table surface to a goal pose.
The gripper stays closed at all times. Uses PyTorch tensors exclusively.

Scene layout (from model.robot_config.SceneProfile):
  - Table: Box at (0,0,0.2), size (0.4,0.5,0.4), top surface at Z=0.4
  - Robot: base at (-0.4, 0.0, 0.4), 180° yaw around Z
  - Object: cube on table surface, XY in [-0.16,0.16]×[-0.20,0.20]
  - Goal: XY in [-0.16,0.16]×[-0.20,0.20]

Control architecture (18D action space):
  - delta_pose [0:6]: task-space error → DLS IK → joint-space delta
  - kp_raw [6:12]: adaptive proportional gains mapped from [-1,1] to physical range
  - kd_raw [12:18]: adaptive derivative gains mapped from [-1,1] to physical range

Observation vector (32D):
  [0:3]   EE position (local XYZ)
  [3:7]   EE quaternion (local wxyz)
  [7:10]  Cube position (local XYZ)
  [10:12] Goal position (local XY)
  [12:15] EE→Cube vector (local XYZ)
  [15:17] Cube→Goal vector (local XY)
  [17:23] Arm joint positions (6)
  [23:29] Arm joint velocities (6)
  [29:32] Point cloud centroid (local XYZ)

Tuning rationale (Tuned 2026-06 based on diagnostic analysis):
  - episode_length_s=7.0: gives agent enough steps to complete the task
  - PD gains (low-gain compliant control):
      kp=[10,20,20,10,5,5], kd=[2,2,2,1,1,0.5]
      Low kp allows compliant, natural motion; kd provides damping.
  - force_range: [24,24,24,9,9,5] Nm — matches low-gain PD, prevents
    excessive force while allowing sufficient torque for pushing.

External dependencies (used as-is, no modifications):
  - genesis: Full simulator implementation (production-grade)
  - torch: PyTorch tensor operations
  - tensordict: TensorDict for rsl_rl 4.x observation interface
"""

import math
import torch
from tensordict import TensorDict

import genesis as gs

from model.robot_config import (
    URDF_PATH,
    JOINT_NAMES,
    ARM_DOF,
    GRIPPER_DOF,
    TOTAL_DOF,
    EE_LINK_NAME,
    GRIPPER_LEFT_LINK,
    GRIPPER_RIGHT_LINK,
    CONTROL_KP,
    CONTROL_KV,
    FORCE_LIMITS_LOWER,
    FORCE_LIMITS_UPPER,
    RL_PUSH_DEFAULT_POSITION,
    get_rl_push_profile,
    get_rl_push_scene_profile,
    get_rl_push_object_randomizer,
    get_rl_push_success_config,
    ObjectRandomizer,
)
from model.pointcloud import (
    PointCloudProcessor,
    PointCloudConfig,
    get_rl_push_pointcloud_config,
    transform_pcd_by_pose,
    farthest_point_sample,
)
from controller.ik_solver import DLSSolver
from controller.task_space_controller import TaskSpaceController


class PushEnv:
    """
    RL environment for non-prehensile pushing with YHRG S1 arm.

    The robot must push a cube on the ground surface to a randomly sampled
    goal position. The gripper remains closed throughout.

    Reward, termination, randomization, and success logic migrated from
    IsaacGym AbbPushBox (s2_cyclic_geometry_obs.py).

    Compatible with rsl_rl 4.x OnPolicyRunner / VecEnv interface.
    """

    # ── YHRG S1 robot constants (from model.robot_config) ──
    URDF_PATH = URDF_PATH
    JOINT_NAMES = JOINT_NAMES
    ARM_DOF = ARM_DOF
    GRIPPER_DOF = GRIPPER_DOF
    TOTAL_DOF = TOTAL_DOF
    EE_LINK_NAME = EE_LINK_NAME
    GRIPPER_LEFT_LINK = GRIPPER_LEFT_LINK
    GRIPPER_RIGHT_LINK = GRIPPER_RIGHT_LINK

    # ── Success & termination thresholds (from SuccessConfig) ──
    # SUCCESS_THRESHOLD is now read from self.success_config.dist_threshold

    # ── Environment grid spacing (visualization) ──
    ENV_SPACING = (2.0, 2.0)  # 2m between adjacent environments

    def __init__(
        self,
        num_envs: int = 16,
        show_viewer: bool = False,
        ctrl_dt: float = 0.01,
        episode_length_s: float = 5.0,
        action_scale: float = 0.30,
        object_randomizer: ObjectRandomizer = None,
    ) -> None:
        self.num_envs = num_envs
        self.num_obs = 32
        self.num_actions = 18
        self.device = gs.device

        self.ctrl_dt = ctrl_dt
        self.action_scale = action_scale
        self.max_episode_length = math.ceil(episode_length_s / ctrl_dt)

        # rsl_rl 4.x expects env.cfg
        self.cfg = {
            "num_envs": num_envs,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "episode_length_s": episode_length_s,
            "ctrl_dt": ctrl_dt,
        }

        # ── Scene profile (from Model layer) ──
        self.scene_profile = get_rl_push_scene_profile()
        sp = self.scene_profile

        # ── Success config (from Model layer) ──
        self.success_config = get_rl_push_success_config()

        # ── build scene ──
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(0.5 / ctrl_dt),
            ),
            show_viewer=show_viewer,
        )

        # ground plane
        self.scene.add_entity(gs.morphs.Plane(fixed=True))

        # table (from SceneProfile)
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                pos=sp.table_pos,
                size=sp.table_dims,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.4, 0.2)),
            ),
        )

        # robot (from SceneProfile: positioned on table top, 180° yaw)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.URDF_PATH,
                pos=sp.robot_base_pos,
                quat=sp.robot_base_quat,
                fixed=True,
            ),
        )

        # push target cube (dynamic, spawned on table surface)
        # ── Object randomizer ──
        self.obj_randomizer = object_randomizer or get_rl_push_object_randomizer()
        sampled_sizes = self.obj_randomizer.sample_sizes(num_envs)

        # Z height: use max possible size to ensure no penetration
        obj_z = self.obj_randomizer.get_max_obj_z(sp.table_top_z, sp.z_eps)

        # Check if heterogeneous entity is needed (multiple distinct sizes)
        unique_sizes = set(sampled_sizes)
        if self.obj_randomizer.size_enabled and len(unique_sizes) > 1:
            # Heterogeneous entity: pass morph list for per-env size variation
            morphs = [gs.morphs.Box(size=s, pos=(0.0, 0.0, obj_z), fixed=False) for s in sampled_sizes]
            self.cube = self.scene.add_entity(
                morphs,
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.2, 0.8, 0.2)),
                ),
            )
        else:
            # Single size (original logic)
            self.cube = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0.0, 0.0, obj_z),
                    size=sp.cube_size,
                    fixed=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.2, 0.8, 0.2)),
                ),
            )

        # Goal marker (visualization only): thin red square on the table top.
        # fixed=True + collision=False -> pure visual entity, never affects physics.
        # Kept in the same add_entity flow as the cube so it is replicated across
        # all parallel envs by scene.build() below.
        self.goal_entity = self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, sp.goal_marker_z),
                size=(sp.goal_marker_size, sp.goal_marker_size, sp.goal_marker_thickness),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=sp.goal_marker_color),
            ),
        )

        # Store per-env sizes and randomization state
        self._sampled_sizes = sampled_sizes
        self._sampled_masses = None
        self._sampled_frictions = None

        # build with parallel envs on rectangular grid (2m spacing)
        self.scene.build(n_envs=num_envs, env_spacing=self.ENV_SPACING)

        # ── DOF indices ──
        self.motor_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.JOINT_NAMES],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.arm_dof_idx = self.motor_dof_idx[:self.ARM_DOF]
        self.gripper_dof_idx = self.motor_dof_idx[self.ARM_DOF:]

        # links - gripper fingers for EE position calculation
        self._ee_link = self.robot.get_link(self.EE_LINK_NAME)  # kept for reference
        self._gripper_left = self.robot.get_link(self.GRIPPER_LEFT_LINK)
        self._gripper_right = self.robot.get_link(self.GRIPPER_RIGHT_LINK)

        # ── Force limits - unified from model.robot_config ──
        profile = get_rl_push_profile()
        self.robot.set_dofs_force_range(
            torch.tensor(profile.force_lower, dtype=gs.tc_float, device=gs.device),
            torch.tensor(profile.force_upper, dtype=gs.tc_float, device=gs.device),
        )

        # ── Set base friction on cube (for friction ratio randomization) ──
        if self.obj_randomizer.friction_enabled:
            self.cube.set_friction(self.obj_randomizer.base_friction)

        # ── Task-space controller (DLS IK + Adaptive PD) ──
        self.task_ctrl = TaskSpaceController(
            robot=self.robot,
            ee_link=self._ee_link,
            arm_dof_idx=self.arm_dof_idx,
            gripper_dof_idx=self.gripper_dof_idx,
            ik_solver=DLSSolver(damping=0.01),
            delta_pos_scale=0.05,
            delta_orient_scale=0.3,
            kp_range=(profile.kp[:ARM_DOF] * 0.5, profile.kp[:ARM_DOF] * 2.0),
            kd_range=(profile.kv[:ARM_DOF] * 0.5, profile.kv[:ARM_DOF] * 2.0),
            gripper_kp=400.0,
            gripper_kd=40.0,
        )
        # Set initial PD gains to mid-range values
        self.task_ctrl.set_initial_gains(num_envs)

        # ── Point cloud processor ──
        self.pc_config = get_rl_push_pointcloud_config()
        self.pc_processor = PointCloudProcessor(self.pc_config)

        # Per-env object sizes tensor for point cloud sampling
        # Initialize from sampled_sizes (set during object creation above)
        self._obj_sizes = torch.tensor(
            self._sampled_sizes, dtype=gs.tc_float, device=gs.device
        )  # [n_envs, 3]

        # Current point cloud buffer
        self._current_pcd = torch.zeros(
            (num_envs, self.pc_config.n_points, 3), dtype=gs.tc_float, device=gs.device
        )

        # Goal point cloud buffer (world frame, generated at reset)
        self._goal_pcd = torch.zeros_like(self._current_pcd)

        # FPS-downsampled point clouds for ICP (precomputed at reset)
        self._fps_pcd = torch.zeros(
            (num_envs, self.pc_config.icp_n_points, 3), dtype=gs.tc_float, device=gs.device
        )
        self._fps_goal_pcd = torch.zeros_like(self._fps_pcd)

        # Cache for _is_success() result (avoid double computation per step)
        self._cached_is_success = None

        # default joint angles — from model.robot_config (RL_PUSH_DEFAULT_POSITION)
        # j1=-0.053, j2=2.079, j3=0.974, j4=0.429 → EE at (0.294, -0.015, 0.093)
        # This places EE in the center of cube workspace (X=0.2-0.4, Y=-0.2-0.2)
        self.default_dof_pos = torch.tensor(
            RL_PUSH_DEFAULT_POSITION,
            dtype=gs.tc_float,
            device=gs.device,
        )

        # ── Move bounds tensors to device (from SceneProfile) ──
        self.obj_bounds_low = torch.tensor(sp.obj_bounds_low, device=gs.device)
        self.obj_bounds_high = torch.tensor(sp.obj_bounds_high, device=gs.device)
        self.obj_pos_low = torch.tensor(sp.obj_xy_low, device=gs.device)
        self.obj_pos_high = torch.tensor(sp.obj_xy_high, device=gs.device)
        self.goal_pos_low = torch.tensor(sp.goal_xy_low, device=gs.device)
        self.goal_pos_high = torch.tensor(sp.goal_xy_high, device=gs.device)

        # ── Environment origins (for local↔world coordinate conversion) ──
        # Genesis: get_pos() returns local coords (excludes envs_offset).
        # envs_offset is the visual offset of each env on the grid.
        # Store as tensor for monitoring: world_pos = local_pos + env_origins
        self.env_origins = torch.tensor(
            self.scene.envs_offset, dtype=gs.tc_float, device=gs.device
        )

        # ── buffers ──
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.time_out_buf = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)

        # goal positions (XY on ground surface)
        self.goal_pos_xy = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # previous cube-to-goal distance for velocity-based push reward
        self.prev_cube_goal_dist = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # success tracking
        self.success_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.extras = dict()

        # episode reward sums for logging
        self.episode_sums = {
            "approach_reward": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "push_reward": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "dist_reward": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "success_reward": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "action_penalty": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
        }

    # ────────────────────── reset ──────────────────────

    def reset(self) -> tuple:
        """Full reset of all environments."""
        self.reset_buf[:] = True
        self._reset_idx(self.reset_buf)
        self.scene.step()
        self._compute_obs()
        return self.get_observations()

    def _reset_idx(self, envs_idx) -> None:
        """Reset selected environments."""
        num_reset = envs_idx.sum().item() if envs_idx.dtype == torch.bool else len(envs_idx)
        if num_reset == 0:
            self.extras["log"] = {}
            return

        # ── Compute per-episode stats BEFORE resetting ──
        log_dict = {}

        if envs_idx.dtype == torch.bool:
            success_rate = self.success_buf[envs_idx].mean()
        else:
            success_rate = self.success_buf[envs_idx].mean() if num_reset > 0 else torch.tensor(0.0, device=self.device)
        log_dict["/success_rate"] = success_rate.item()

        for key, value in self.episode_sums.items():
            if envs_idx.dtype == torch.bool:
                n = envs_idx.sum()
                mean = torch.where(n > 0, value[envs_idx].sum() / n, torch.tensor(0.0, device=self.device))
            else:
                mean = value[envs_idx].mean() if num_reset > 0 else torch.tensor(0.0, device=self.device)
            log_dict["/" + key] = mean.item()
        
        self.extras["log"] = log_dict

        # ── Reset robot ──
        self.robot.set_qpos(self.default_dof_pos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # ── Randomise push object position ──
        # NOTE: Genesis' zero-copy fast path (bool envs_idx) requires the input
        # tensor to have shape [num_envs, *] (full batch), not [num_reset, *].
        # Always build a full-batch tensor and only fill the selected envs.
        is_bool_idx = envs_idx.dtype == torch.bool
        rand_xy = torch.rand(num_reset, 2, device=self.device)
        obj_xy = self.obj_pos_low + rand_xy * (self.obj_pos_high - self.obj_pos_low)
        # Z height: use max possible size to ensure no penetration for all variants
        obj_z_val = self.obj_randomizer.get_max_obj_z(self.scene_profile.table_top_z, self.scene_profile.z_eps)
        obj_z = torch.full((num_reset, 1), obj_z_val, device=self.device)
        cube_pos_subset = torch.cat([obj_xy, obj_z], dim=-1)
        if is_bool_idx:
            cube_pos = torch.empty(self.num_envs, 3, device=self.device)
            cube_pos[envs_idx] = cube_pos_subset
        else:
            cube_pos = cube_pos_subset
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)

        # Random Z-axis orientation (yaw range from SceneProfile: [-90°, 90°])
        rand_yaw = (torch.rand(num_reset, device=self.device) * 2 - 1) * self.scene_profile.obj_yaw_range
        half_yaw = rand_yaw * 0.5
        cube_quat_subset = torch.stack([
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ], dim=-1)
        if is_bool_idx:
            cube_quat = torch.zeros(self.num_envs, 4, device=self.device)
            cube_quat[envs_idx] = cube_quat_subset
        else:
            cube_quat = cube_quat_subset
        self.cube.set_quat(cube_quat, envs_idx=envs_idx)

        # ── Mass randomization ──
        mass_tensor = self.obj_randomizer.sample_masses(num_reset, self.device)
        if mass_tensor is not None:
            # Build full-batch tensor for Genesis set_* (see cube_pos note above)
            if is_bool_idx:
                mass_2d = torch.zeros(self.num_envs, 1, device=self.device)
                mass_2d[envs_idx] = mass_tensor.unsqueeze(-1)
            else:
                mass_2d = mass_tensor.unsqueeze(-1)  # (num_reset, 1) — single-link entity
            self.cube.set_links_inertial_mass(mass_2d, links_idx_local=0, envs_idx=envs_idx)
            # Store for query interface
            if self._sampled_masses is None:
                self._sampled_masses = torch.zeros(self.num_envs, device=self.device)
            if envs_idx.dtype == torch.bool:
                self._sampled_masses[envs_idx] = mass_tensor
            else:
                self._sampled_masses[envs_idx] = mass_tensor

        # ── Friction randomization ──
        friction_ratio = self.obj_randomizer.sample_friction_ratios(num_reset, self.device)
        if friction_ratio is not None:
            # Build full-batch tensor for Genesis set_* (see cube_pos note above)
            if is_bool_idx:
                ratio_2d = torch.zeros(self.num_envs, 1, device=self.device)
                ratio_2d[envs_idx] = friction_ratio.unsqueeze(-1)
            else:
                ratio_2d = friction_ratio.unsqueeze(-1)  # (num_reset, 1) — single-link entity
            self.cube.set_friction_ratio(ratio_2d, links_idx_local=0, envs_idx=envs_idx)
            # Store absolute friction for query interface
            if self._sampled_frictions is None:
                self._sampled_frictions = torch.zeros(self.num_envs, device=self.device)
            abs_friction = friction_ratio * self.obj_randomizer.base_friction
            if envs_idx.dtype == torch.bool:
                self._sampled_frictions[envs_idx] = abs_friction
            else:
                self._sampled_frictions[envs_idx] = abs_friction

        # ── Randomise goal position ──
        # Ensure goal is at least MIN_INIT_DIST away from cube to avoid trivial success
        MIN_INIT_DIST = self.success_config.dist_threshold + 0.03  # > dist_threshold so episode doesn't start already done
        # Vectorized rejection sampling: generate candidates and keep valid ones
        goal_range = self.goal_pos_high - self.goal_pos_low
        new_goal_xy = self.goal_pos_low + torch.rand(num_reset, 2, device=self.device) * goal_range
        dist = torch.linalg.norm(new_goal_xy - obj_xy, dim=-1)
        invalid = dist < MIN_INIT_DIST
        # Retry invalid entries (up to 5 rounds of vectorized rejection)
        for _ in range(5):
            if not invalid.any():
                break
            n_invalid = invalid.sum()
            retry = self.goal_pos_low + torch.rand(n_invalid, 2, device=self.device) * goal_range
            new_goal_xy[invalid] = retry
            dist = torch.linalg.norm(new_goal_xy - obj_xy, dim=-1)
            invalid = dist < MIN_INIT_DIST
        if envs_idx.dtype == torch.bool:
            self.goal_pos_xy[envs_idx] = new_goal_xy
        else:
            self.goal_pos_xy[envs_idx] = new_goal_xy

        # Goal yaw (currently fixed at 0). To randomize the goal orientation,
        # sample it here and it will be picked up by both the marker rotation
        # below and the goal point-cloud generation.
        goal_yaw = torch.zeros(num_reset, device=self.device)

        # ── Sync goal marker (visualization only) ──
        # Same full-batch set_pos/set_quat pattern as the cube (Genesis'
        # zero-copy fast path requires [num_envs, *] shapes for bool masks).
        goal_z_val = self.scene_profile.goal_marker_z
        goal_pos_subset = torch.cat([
            new_goal_xy,
            torch.full((num_reset, 1), goal_z_val, device=self.device),
        ], dim=-1)
        if is_bool_idx:
            goal_pos = torch.empty(self.num_envs, 3, device=self.device)
            goal_pos[envs_idx] = goal_pos_subset
        else:
            goal_pos = goal_pos_subset
        self.goal_entity.set_pos(goal_pos, envs_idx=envs_idx)

        half_yaw = goal_yaw * 0.5
        goal_quat_subset = torch.stack([
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ], dim=-1)
        if is_bool_idx:
            goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            goal_quat[envs_idx] = goal_quat_subset
        else:
            goal_quat = goal_quat_subset
        self.goal_entity.set_quat(goal_quat, envs_idx=envs_idx)

        # ── Regenerate point cloud for reset envs ──
        if self.pc_config.enabled:
            reset_sizes = self._obj_sizes[envs_idx] if envs_idx.dtype != torch.bool else self._obj_sizes[envs_idx]
            new_pcd = self.pc_processor.generate_point_cloud(reset_sizes, num_reset)
            if envs_idx.dtype == torch.bool:
                self._current_pcd[envs_idx] = new_pcd
            else:
                self._current_pcd[envs_idx] = new_pcd

            # Generate goal point cloud (world frame)
            new_goal_pcd = self.pc_processor.generate_goal_point_cloud(
                new_pcd, new_goal_xy, goal_yaw
            )
            if envs_idx.dtype == torch.bool:
                self._goal_pcd[envs_idx] = new_goal_pcd
            else:
                self._goal_pcd[envs_idx] = new_goal_pcd

            # FPS downsample for ICP (precompute at reset)
            new_fps_pcd = farthest_point_sample(new_pcd, self.pc_config.icp_n_points)
            new_fps_goal_pcd = farthest_point_sample(new_goal_pcd, self.pc_config.icp_n_points)
            if envs_idx.dtype == torch.bool:
                self._fps_pcd[envs_idx] = new_fps_pcd
                self._fps_goal_pcd[envs_idx] = new_fps_goal_pcd
            else:
                self._fps_pcd[envs_idx] = new_fps_pcd
                self._fps_goal_pcd[envs_idx] = new_fps_goal_pcd

        # ── Reset episode buffers ──
        if envs_idx.dtype == torch.bool:
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.actions.masked_fill_(envs_idx.unsqueeze(-1), 0.0)
            self.success_buf.masked_fill_(envs_idx, 0.0)
            self.prev_cube_goal_dist.masked_fill_(envs_idx, 0.0)
            for v in self.episode_sums.values():
                v.masked_fill_(envs_idx, 0.0)
        else:
            self.episode_length_buf[envs_idx] = 0
            self.actions[envs_idx] = 0.0
            self.success_buf[envs_idx] = 0.0
            self.prev_cube_goal_dist[envs_idx] = 0.0
            for v in self.episode_sums.values():
                v[envs_idx] = 0.0

    # ────────────────────── step ──────────────────────

    def step(self, actions: torch.Tensor) -> tuple:
        """Run one environment step."""
        self._cached_is_success = None  # Reset cache for new step
        self.actions = actions.clamp(-1.0, 1.0)
        self.episode_length_buf += 1

        # ── Parse 18D actions: [delta_pose(6), kp_raw(6), kd_raw(6)] ──
        delta_pose = self.actions[:, :6]
        kp_raw = self.actions[:, 6:12]
        kd_raw = self.actions[:, 12:18]

        # ── Task-space control: DLS IK + Adaptive PD ──
        self.task_ctrl.compute_and_apply(delta_pose, kp_raw, kd_raw)

        # Step physics
        self.scene.step()

        # ── Compute rewards (IsaacGym L1003-1033) ──
        self._compute_rewards()

        # ── Compute termination (IsaacGym L985-1001) ──
        self._compute_termination()

        # Reset done envs
        self._reset_idx(self.reset_buf)

        # ── Compute observations ──
        self._compute_obs()
        obs_td = self.get_observations()

        return obs_td, self.rew_buf, self.reset_buf, self.extras

    # ────────────────────── rewards ──────────────────────

    def _compute_rewards(self) -> None:
        """Compute all reward components.

        Phased reward structure designed for effective push learning:

          1. approach_reward: dense EE→cube distance. Uses exp(-d/0.3) with
             wide sigma so there's always gradient even when EE is far.
             Weight: 0.01 (per-step). With ~500 steps/episode, accumulates
             to ~5.0, balanced with success_reward (~1.0) and dist_reward
             (~10.0). Same order of magnitude across the three signals.

          2. push_reward: velocity-based reward for cube moving toward goal.
             Positive when cube-goal distance decreases, negative when it
             increases. This is the KEY signal that teaches pushing.
             Weight: 5.0 (dominant to emphasize pushing). Clamped to [-1,1].

          3. dist_reward: dense cube→goal distance. Uses exp(-d/0.15).
             UNGATED — doesn't depend on EE position.
             Weight: 0.02 (per-step). Accumulates to ~10 per episode,
             matched to success_reward order of magnitude.

          4. success_reward: sparse bonus on task completion.
             Weight: +10 per step while succeeded. Accumulates to 0~10
             depending on fraction of episode spent in success state.

          5. action_penalty: small L2 penalty on actions.
             Weight: -0.005 (per-step). Accumulates to ~-2.5 per episode.

        Per-episode accumulation order of magnitude (target 1~10 each):
          approach_reward ~5    dist_reward ~10    success_reward ~1
        """
        cube_pos = self.cube.get_pos()
        cube_xy = cube_pos[:, :2]
        goal_xy = self.goal_pos_xy
        ee_pos = self._get_ee_pos()

        # 3D distance for approach (EE is 30cm above cube — need Z gradient!)
        ee_obj_dist_3d = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
        cube_goal_dist = torch.linalg.norm(goal_xy - cube_xy, dim=-1)

        # 1. Approach reward: EE → cube in 3D (wider sigma for large Z gap)
        #    Weight 0.01 keeps it in same order-of-magnitude as success/dist.
        approach_rew = torch.exp(-ee_obj_dist_3d / 0.3) * 0.01 - 0.005

        # 2. Push reward: velocity-based, cube moving toward goal
        #    Positive when cube_goal_dist decreases (i.e. cube moves toward goal)
        #    On first step of episode (prev_dist=0), use current dist as baseline
        first_step = (self.episode_length_buf == 1)
        prev_dist = torch.where(first_step, cube_goal_dist, self.prev_cube_goal_dist)
        push_rew = (prev_dist - cube_goal_dist) * 0.10  # positive when approaching
        push_rew = push_rew.clamp(-1.0, 1.0)  # clip to avoid instability

        # 3. Distance reward: cube → goal (ungated, wide sigma)
        #    Weight 0.02 keeps it in same order-of-magnitude as success.
        dist_rew = torch.exp(-cube_goal_dist / 0.15) * 0.02 -0.0060

        # 4. Success reward: sparse bonus
        is_succ = self._is_success()
        self._cached_is_success = is_succ  # Cache for _compute_termination
        success_rew = is_succ * 10.0

        # 5. Action penalty
        action_pen = -0.0005 * torch.sum(self.actions ** 2, dim=-1)

        # Total reward
        self.rew_buf = approach_rew + push_rew + dist_rew + success_rew + action_pen

        # Save current distance for next step's velocity reward
        self.prev_cube_goal_dist = cube_goal_dist.detach().clone()

        # Accumulate for logging
        self.episode_sums["approach_reward"] += approach_rew
        self.episode_sums["push_reward"] += push_rew
        self.episode_sums["dist_reward"] += dist_rew
        self.episode_sums["success_reward"] += success_rew
        self.episode_sums["action_penalty"] += action_pen

        # Update per-episode success flag
        self.success_buf = torch.max(self.success_buf, is_succ)

    # ────────────────────── termination ──────────────────────

    def _get_ee_pos(self) -> torch.Tensor:
        """Get EE position using wrist link (6_Link).
        
        Note: Gripper fingers (7_Link, 8_Link) are mounted below the wrist,
        so their Z coordinates are negative. Using wrist position is more
        appropriate for pushing tasks.
        """
        return self._ee_link.get_pos()

    def _get_ee_quat(self) -> torch.Tensor:
        """Get EE quaternion using wrist link."""
        return self._ee_link.get_quat()

    def _compute_termination(self) -> None:
        """Compute termination conditions.
        
        Termination triggers on:
          1. Timeout (episode_length > max_episode_length)
          2. Object out of bounds (cube XY exits workspace)
          3. Success (cube reaches goal)
        
        Note: EE out-of-bounds termination is intentionally removed.
        The arm should be free to explore without premature termination.
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        cube_pos = self.cube.get_pos()
        cube_xy = cube_pos[:, :2]
        obj_outbound = (
            torch.any(cube_xy < self.obj_bounds_low, dim=1) |
            torch.any(cube_xy > self.obj_bounds_high, dim=1)
        )

        if self._cached_is_success is not None:
            is_succ = self._cached_is_success.to(torch.bool)
            self._cached_is_success = None  # Clear cache
        else:
            is_succ = self._is_success().to(torch.bool)
        self.last_success = is_succ  # per-env success state (readable after _reset_idx clears success_buf)
        self.reset_buf = self.time_out_buf | obj_outbound | is_succ
        self.extras["time_outs"] = self.time_out_buf.to(dtype=gs.tc_float)

    # ────────────────────── success ──────────────────────

    def _is_success(self) -> torch.Tensor:
        """Success criterion: distance + velocity + ICP alignment (three conditions).

        Condition 1: object-to-goal XY distance < dist_threshold
        Condition 2: object XY linear velocity < vel_threshold (if enabled)
        Condition 3: ICP alignment error < icp_threshold (if enabled)
        """
        cfg = self.success_config
        cube_xy = self.cube.get_pos()[:, :2]
        obj_goal_dist = torch.linalg.norm(self.goal_pos_xy - cube_xy, dim=-1)
        dist_ok = obj_goal_dist < cfg.dist_threshold

        if cfg.vel_enabled:
            cube_vel = self.cube.get_vel()[:, :2]  # XY线速度
            cube_speed = torch.linalg.norm(cube_vel, dim=-1)
            vel_ok = cube_speed < cfg.vel_threshold
        else:
            vel_ok = torch.ones_like(dist_ok)

        if cfg.icp_enabled and self.pc_config.enabled:
            # Transform FPS-downsampled local pcd to current world frame
            cube_pos = self.cube.get_pos()
            cube_yaw = self._get_cube_yaw()
            current_world_pcd = transform_pcd_by_pose(
                self._fps_pcd, cube_pos[:, :2], cube_yaw, cube_pos[:, 2]
            )
            icp_error = self.pc_processor.compute_icp_error_downsampled(
                current_world_pcd, self._fps_goal_pcd
            )
            icp_ok = icp_error < cfg.icp_threshold
        else:
            icp_ok = torch.ones_like(dist_ok)

        return (dist_ok & vel_ok & icp_ok).to(gs.tc_float)

    def _get_cube_yaw(self) -> torch.Tensor:
        """Extract the Z-axis yaw angle from the object quaternion."""
        quat = self.cube.get_quat()  # [n_envs, 4] (w, x, y, z)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    # ────────────────────── observations ──────────────────────

    def _compute_obs(self) -> None:
        """Observation vector (dim = 32), all in environment-local coordinates.

        Genesis guarantees: get_pos() returns local coordinates (excludes envs_offset).
        All positions are relative to each environment's own origin (robot base).
        This ensures the RL policy is invariant to environment placement on the grid.

        Observation breakdown:
          [0:3]   EE position (local XYZ)
          [3:7]   EE quaternion (local wxyz)
          [7:10]  Cube position (local XYZ)
          [10:12] Goal position (local XY)
          [12:15] EE→Cube vector (local XYZ)
          [15:17] Cube→Goal vector (local XY)
          [17:23] Arm joint positions (6)
          [23:29] Arm joint velocities (6)
          [29:32] Point cloud centroid (local XYZ)
        """
        ee_pos = self._get_ee_pos()       # local coords
        ee_quat = self._get_ee_quat()     # local coords
        cube_pos = self.cube.get_pos()    # local coords
        ee_to_cube = cube_pos - ee_pos
        cube_to_goal = self.goal_pos_xy - cube_pos[:, :2]

        # Joint states
        qpos = self.robot.get_qpos()[:, :self.ARM_DOF]
        qvel = self.robot.get_dofs_velocity()[:, :self.ARM_DOF]

        # Point cloud centroid
        if self.pc_config.enabled:
            centroid = PointCloudProcessor.compute_centroid(self._current_pcd)  # [n_envs, 3]
        else:
            centroid = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device)

        # RL observation vector
        self.obs_buf = torch.cat([
            ee_pos,            # 3  — local
            ee_quat,           # 4  — local
            cube_pos,          # 3  — local
            self.goal_pos_xy,  # 2  — local (sampled in local ranges)
            ee_to_cube,        # 3  — local
            cube_to_goal,      # 2  — local
            qpos,              # 6  — joint positions
            qvel,              # 6  — joint velocities
            centroid,          # 3  — point cloud centroid
        ], dim=-1)

    # ────────────────────── rsl_rl 4.x interface ──────────────────────

    def get_observations(self) -> TensorDict:
        """Return observations as TensorDict expected by rsl_rl 4.x."""
        return TensorDict(
            {"policy": self.obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self) -> None:
        """Clean up simulation resources."""
        pass  # Genesis Scene has no explicit close/stop method

    # ────────────────────── randomization query ──────────────────────

    def get_object_randomization_params(self) -> dict:
        """Query per-env object randomization parameters.

        Returns:
            dict with keys:
                sizes: List[Tuple[float,float,float]] — per-env object sizes
                masses: torch.Tensor or None — per-env mass (n_envs,)
                frictions: torch.Tensor or None — per-env friction coefficient (n_envs,)
                randomizer: ObjectRandomizer — the randomization strategy config
        """
        return {
            "sizes": self._sampled_sizes,
            "masses": self._sampled_masses,
            "frictions": self._sampled_frictions,
            "randomizer": self.obj_randomizer,
        }

    # ────────────────────── point cloud query ──────────────────────

    def get_point_cloud(self) -> torch.Tensor:
        """Get current point cloud for all environments.

        Returns:
            Point cloud [n_envs, n_points, 3] in object-local frame (centered at origin)
        """
        return self._current_pcd.clone()

    def get_goal_point_cloud(self) -> torch.Tensor:
        """Get cached goal point cloud (world frame, generated at reset).

        Returns:
            Goal point cloud [n_envs, n_points, 3]
        """
        if not self.pc_config.enabled:
            return torch.zeros_like(self._current_pcd)
        return self._goal_pcd.clone()

    # ────────────────────── coordinate conversion (monitoring) ──────────────────────

    def local_to_world(self, local_pos: torch.Tensor) -> torch.Tensor:
        """Convert local coordinates to world coordinates (for monitoring).
        
        Args:
            local_pos: (..., 3) tensor in environment-local frame.
        
        Returns:
            (..., 3) tensor in world frame (adds env grid offset).
        """
        return local_pos + self.env_origins

    def world_to_local(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to local coordinates.
        
        Args:
            world_pos: (..., 3) tensor in world frame.
        
        Returns:
            (..., 3) tensor in environment-local frame.
        """
        return world_pos - self.env_origins
