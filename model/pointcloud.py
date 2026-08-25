"""
Object point-cloud processing module.

Provides box-surface uniform sampling, FPS downsampling, ICP registration, etc.,
wrapping different point-cloud preprocessing algorithms with the Strategy
pattern to ensure extensibility.

Core components:
  - BoxSurfaceSampler: area-weighted uniform sampling of the box surface
  - FarthestPointSampler: farthest-point-sampling (FPS) downsampling
  - ICPAligner: iterative closest point (ICP) registration
  - PointCloudProcessor: facade that orchestrates sampling + FPS + ICP
  - ISamplingStrategy / UniformSamplingStrategy: sampling strategy interfaces

Design patterns:
  - Strategy: ISamplingStrategy encapsulates the sampling algorithm
  - Facade: PointCloudProcessor orchestrates the full point-cloud pipeline
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from pytorch3d.ops import knn_points


# ── Sampling strategy interface (Strategy Pattern) ──

class ISamplingStrategy(ABC):
    """Abstract interface for point-cloud sampling strategies."""

    @abstractmethod
    def sample(self, n_envs: int, n_points: int, sizes: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        """Generate a point cloud.

        Args:
            n_envs: number of parallel environments
            n_points: number of sampled points per environment
            sizes: object sizes [n_envs, 3] (full dimensions, not half-extents)
            device: computation device

        Returns:
            point cloud [n_envs, n_points, 3] in object-local coordinates
            (centered at origin)
        """
        ...


class UniformSamplingStrategy(ISamplingStrategy):
    """Box-surface area-weighted uniform sampling strategy.

    Algorithm:
      1. Generate the 6 faces of a unit box (4 vertices per face)
      2. Select faces with area-weighted random sampling
      3. Sample points uniformly on the chosen face via barycentric coords
      4. Scale by the object's actual dimensions
    """

    def sample(self, n_envs: int, n_points: int, sizes: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        return box_surface_sample(n_envs, n_points, sizes, device)


# ── Box surface sampling ──

def _get_unit_box_face_verts(device: torch.device) -> torch.Tensor:
    """Generate the 6 face vertices of a unit box [-0.5, 0.5]^3.

    Returns:
        face_verts [6, 4, 3] — 3D coordinates of the 4 vertices per face
    """
    # 8 vertices
    v = torch.tensor([
        [-0.5, -0.5, -0.5],  # 0
        [+0.5, -0.5, -0.5],  # 1
        [+0.5, +0.5, -0.5],  # 2
        [-0.5, +0.5, -0.5],  # 3
        [-0.5, -0.5, +0.5],  # 4
        [+0.5, -0.5, +0.5],  # 5
        [+0.5, +0.5, +0.5],  # 6
        [-0.5, +0.5, +0.5],  # 7
    ], dtype=torch.float32, device=device)

    # 6 faces (vertex order guarantees outward normals)
    faces = torch.tensor([
        [0, 1, 2, 3],  # -Z (bottom)
        [4, 7, 6, 5],  # +Z (top)
        [0, 4, 5, 1],  # -Y (front)
        [2, 6, 7, 3],  # +Y (back)
        [0, 3, 7, 4],  # -X (left)
        [1, 5, 6, 2],  # +X (right)
    ], dtype=torch.long, device=device)

    return v[faces]  # [6, 4, 3]


def box_surface_sample(
    n_envs: int,
    n_points: int,
    sizes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Area-weighted uniform sampling of the box surface.

    Args:
        n_envs: number of parallel environments
        n_points: number of sampled points per environment
        sizes: object full dimensions [n_envs, 3]
        device: computation device

    Returns:
        point cloud [n_envs, n_points, 3] in object-local coordinates
        (centered at origin)
    """
    face_verts = _get_unit_box_face_verts(device)  # [6, 4, 3]

    # Compute the area of each face (unit box)
    # Each face is a rectangle, area = product of its two edge lengths
    # For a unit box all areas are 1.0, but they differ after scaling by sizes
    # area = size_dim1 * size_dim2
    # face0(-Z), face1(+Z): area = sizes[:, 0] * sizes[:, 1]
    # face2(-Y), face3(+Y): area = sizes[:, 0] * sizes[:, 2]
    # face4(-X), face5(+X): area = sizes[:, 1] * sizes[:, 2]
    areas = torch.stack([
        sizes[:, 0] * sizes[:, 1],  # -Z, +Z
        sizes[:, 0] * sizes[:, 1],
        sizes[:, 0] * sizes[:, 2],  # -Y, +Y
        sizes[:, 0] * sizes[:, 2],
        sizes[:, 1] * sizes[:, 2],  # -X, +X
        sizes[:, 1] * sizes[:, 2],
    ], dim=-1)  # [n_envs, 6]

    # Area-weighted probabilities
    face_probs = areas / areas.sum(dim=-1, keepdim=True)  # [n_envs, 6]

    # Select faces with probability
    face_indices = torch.multinomial(face_probs, n_points, replacement=True)  # [n_envs, n_points]

    # Get vertices of the selected faces
    # face_verts: [6, 4, 3] → vertices of selected faces [n_envs, n_points, 4, 3]
    selected_faces = face_verts[face_indices]  # [n_envs, n_points, 4, 3]

    # Uniform triangle sampling: split the quad into 2 triangles
    # P = (1 - sqrt(r1)) * A + sqrt(r1) * (1 - r2) * B + sqrt(r1) * r2 * C
    r1 = torch.rand(n_envs, n_points, 1, device=device)
    r2 = torch.rand(n_envs, n_points, 1, device=device)
    sqrt_r1 = torch.sqrt(r1)

    # Randomly pick which triangle of the quad (split along a diagonal)
    tri_choice = torch.rand(n_envs, n_points, 1, device=device) < 0.5

    # Triangle 1: v0, v1, v2; Triangle 2: v0, v2, v3
    v0 = selected_faces[:, :, 0, :]  # [n_envs, n_points, 3]
    v1 = selected_faces[:, :, 1, :]
    v2 = selected_faces[:, :, 2, :]
    v3 = selected_faces[:, :, 3, :]

    # B and C for triangle 1
    B1, C1 = v1 - v0, v2 - v0
    # B and C for triangle 2
    B2, C2 = v2 - v0, v3 - v0

    B = torch.where(tri_choice, B1, B2)
    C = torch.where(tri_choice, C1, C2)

    # Barycentric sampling
    points = v0 + (1 - sqrt_r1) * 0 + sqrt_r1 * (1 - r2) * B + sqrt_r1 * r2 * C

    # Scale to actual dimensions (unit box → actual size)
    # sizes: [n_envs, 3] → [n_envs, 1, 3]
    points = points * sizes.unsqueeze(1)

    return points  # [n_envs, n_points, 3]


# ── FPS downsampling ──

def farthest_point_sample(
    points: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Farthest point sampling (FPS).

    Args:
        points: input point cloud [n_envs, n_points, 3]
        n_samples: target number of downsampled points

    Returns:
        downsampled point cloud [n_envs, n_samples, 3]
    """
    B, N, _ = points.shape
    device = points.device

    if n_samples >= N:
        return points.clone()

    selected_indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    # Randomly pick the first point
    selected_indices[:, 0] = torch.randint(0, N, (B,), device=device)

    # Initialize min distances to infinity
    min_dist = torch.full((B, N), float('inf'), device=device)

    for i in range(1, n_samples):
        # Previously selected point
        last_idx = selected_indices[:, i - 1]  # [B]
        last_pts = points[torch.arange(B, device=device), last_idx]  # [B, 3]

        # Distance from all points to the previously selected point
        dist = torch.cdist(points, last_pts.unsqueeze(1)).squeeze(-1)  # [B, N]

        # Update min distances
        min_dist = torch.minimum(min_dist, dist)

        # Select the farthest point
        selected_indices[:, i] = torch.argmax(min_dist, dim=-1)

    # Gather points by index
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_samples)
    return points[batch_idx, selected_indices]  # [B, n_samples, 3]


# ── ICP registration ──

@torch.compile(mode="reduce-overhead")
def _kabsch_step(src_centered: torch.Tensor, tgt_centered: torch.Tensor) -> torch.Tensor:
    B = src_centered.shape[0]

    # 1. Compute cross-covariance matrix H [B, 3, 3]
    H = src_centered.transpose(-1, -2) @ tgt_centered

    # Extract elements
    H11, H12, H13 = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    H21, H22, H23 = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    H31, H32, H33 = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]

    # 2. Construct 4x4 symmetric matrix N (Horn's method)
    N = torch.stack([
        torch.stack([H11 + H22 + H33, H23 - H32, H31 - H13, H12 - H21], dim=-1),
        torch.stack([H23 - H32, H11 - H22 - H33, H12 + H21, H31 + H13], dim=-1),
        torch.stack([H31 - H13, H12 + H21, -H11 + H22 - H33, H23 + H32], dim=-1),
        torch.stack([H12 - H21, H31 + H13, H23 + H32, -H11 - H22 + H33], dim=-1)
    ], dim=-2)

    # 3. Shift N to guarantee the most positive eigenvalue is strictly dominant
    shift = torch.norm(N, dim=(1, 2), keepdim=True)  # Shape is [B, 1, 1]
    I = torch.eye(4, device=H.device, dtype=H.dtype).unsqueeze(0) # Shape is [1, 4, 4]

    # FIX: Removed .unsqueeze(-1).
    # [B, 1, 1] * [1, 4, 4] correctly broadcasts to [B, 4, 4]
    N_shifted = N + shift * I

    # 4. Power iteration to find the dominant eigenvector (the quaternion)
    q = torch.ones((B, 4, 1), device=H.device, dtype=H.dtype)

    for _ in range(10):
        q = N_shifted @ q
        q = q / torch.norm(q, dim=1, keepdim=True)

    q = q.squeeze(-1)  # [B, 4]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 5. Convert quaternion to 3x3 rotation matrix
    R_iter = torch.stack([
        torch.stack([1.0 - 2.0*qy**2 - 2.0*qz**2, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw], dim=-1),
        torch.stack([2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx**2 - 2.0*qz**2, 2.0*qy*qz - 2.0*qx*qw], dim=-1),
        torch.stack([2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx**2 - 2.0*qy**2], dim=-1)
    ], dim=-2)

    return R_iter

def icp_align(
    source: torch.Tensor,
    target: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, N, _ = source.shape
    device = source.device

    R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    t = torch.zeros(B, 3, device=device)

    transformed = source.clone()
    prev_error = torch.full((B,), float('inf'), device=device)
    
    # Optimization: Pre-allocate batch indices outside the loop
    batch_idx = torch.arange(B, device=device).unsqueeze(1)

    for _ in range(max_iter):
        knn_result = knn_points(transformed, target, K=1)
        nn_idx = knn_result.idx.squeeze(-1)  # [B, N]
        nn_dists_sq = knn_result.dists.squeeze(-1)  
        nn_target = target[batch_idx, nn_idx]  # [B, N, 3]

        error = nn_dists_sq.sqrt().mean(dim=-1)  # [B]

        if torch.max(torch.abs(prev_error - error)) < tol:
            break
        prev_error = error
        
        # Calculate centers
        src_mean = transformed.mean(dim=1, keepdim=True)
        tgt_mean = nn_target.mean(dim=1, keepdim=True)

        src_centered = transformed - src_mean
        tgt_centered = nn_target - tgt_mean

        # Execute compiled, FLOP-reduced Kabsch algorithm
        R_iter = _kabsch_step(src_centered, tgt_centered)

        # Update translation (squeeze means for correct broadcasting)
        src_mean_sq = src_mean.squeeze(1)
        tgt_mean_sq = tgt_mean.squeeze(1)
        t_iter = tgt_mean_sq - (R_iter @ src_mean_sq.unsqueeze(-1)).squeeze(-1)

        # Update global accumulators
        R = R_iter @ R
        t = (R_iter @ t.unsqueeze(-1)).squeeze(-1) + t_iter

        # Re-apply global transform to source to prevent iterative floating-point drift
        transformed = (R @ source.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(-2)
    
    # Final error compute
    knn_result = knn_points(transformed, target, K=1)
    error = knn_result.dists.squeeze(-1).sqrt().mean(dim=-1)

    return R, t, error

# ── 点云位姿变换 ──

def transform_pcd_by_pose(
    pcd: torch.Tensor,
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    pos_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """将局部坐标系点云变换到世界坐标系。

    使用齐次变换矩阵实现批量位姿变换：XY平移 + Z轴旋转 + Z高度。

    Args:
        pcd: 局部坐标系点云 [B, N, 3]
        pos_xy: XY位置 [B, 2]
        yaw: Z轴偏航角 [B]
        pos_z: Z高度 [B]（默认使用pcd均值Z）

    Returns:
        世界坐标系点云 [B, N, 3]
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # 构造齐次变换矩阵
    T = torch.zeros(B, 4, 4, device=device)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    T[:, 0, 0] = cos_yaw
    T[:, 0, 1] = -sin_yaw
    T[:, 1, 0] = sin_yaw
    T[:, 1, 1] = cos_yaw
    T[:, 0, 3] = pos_xy[:, 0]
    T[:, 1, 3] = pos_xy[:, 1]
    if pos_z is not None:
        T[:, 2, 3] = pos_z
    else:
        T[:, 2, 3] = pcd.mean(dim=1)[:, 2]
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0

    # 齐次坐标变换
    ones = torch.ones(B, N, 1, device=device)
    pcd_h = torch.cat([pcd, ones], dim=-1)  # [B, N, 4]
    result = (T.unsqueeze(1) @ pcd_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]

    return result


# ── 目标点云生成 ──

def generate_goal_point_cloud(
    current_pcd: torch.Tensor,
    goal_pos_xy: torch.Tensor,
    goal_yaw: torch.Tensor,
) -> torch.Tensor:
    """通过对当前点云施加目标位姿的齐次变换生成目标点云。

    目标点云 = 当前点云在目标位姿下的位置（仅XY平移 + Z轴旋转）。

    Args:
        current_pcd: 当前点云 [n_envs, n_points, 3]
        goal_pos_xy: 目标位置XY [n_envs, 2]
        goal_yaw: 目标偏航角 [n_envs]

    Returns:
        目标点云 [n_envs, n_points, 3]
    """
    return transform_pcd_by_pose(current_pcd, goal_pos_xy, goal_yaw)


# ── 点云配置 ──

@dataclass
class PointCloudConfig:
    """点云处理配置。

    Attributes:
        enabled: 是否启用点云处理（关闭时质心为零向量，等效原29维观测）
        n_points: 初始采样点数
        n_fps_samples: FPS降采样目标点数（0表示不降采样）
        icp_max_iter: ICP最大迭代次数
        icp_tol: ICP收敛阈值
        icp_n_points: ICP配准用FPS降采样目标点数（0表示不降采样，直接用n_points）
        name: 配置名称
    """
    enabled: bool = True
    n_points: int = 1024
    n_fps_samples: int = 0       # 0 = 不降采样
    icp_max_iter: int = 20
    icp_tol: float = 1e-5
    icp_n_points: int = 64       # ICP配准用降采样点数
    name: str = ""


def get_rl_push_pointcloud_config() -> PointCloudConfig:
    """获取RL推动任务的点云处理配置。"""
    return PointCloudConfig(
        name="rl_push",
        enabled=True,
        n_points=1024,
        n_fps_samples=0,
        icp_max_iter=20,
        icp_tol=1e-5,
        icp_n_points=64,
    )


# ── 点云处理器 (Facade) ──

class PointCloudProcessor:
    """点云处理外观类，编排采样+FPS+ICP的完整流水线。

    设计模式:
      - Facade: 封装采样、降采样、配准的完整流程
      - Strategy: 通过 ISamplingStrategy 支持可替换的采样算法
    """

    def __init__(
        self,
        config: PointCloudConfig,
        sampling_strategy: Optional[ISamplingStrategy] = None,
    ) -> None:
        self.config = config
        self.sampling_strategy = sampling_strategy or UniformSamplingStrategy()

    def generate_point_cloud(
        self,
        sizes: torch.Tensor,
        n_envs: int,
    ) -> torch.Tensor:
        """生成物体表面点云。

        Args:
            sizes: 物体全尺寸 [n_envs, 3]
            n_envs: 并行环境数

        Returns:
            点云 [n_envs, n_points, 3]（如果n_fps_samples>0则为 [n_envs, n_fps_samples, 3]）
        """
        pcd = self.sampling_strategy.sample(n_envs, self.config.n_points, sizes, sizes.device)

        if self.config.n_fps_samples > 0 and self.config.n_fps_samples < self.config.n_points:
            pcd = farthest_point_sample(pcd, self.config.n_fps_samples)

        return pcd

    def generate_goal_point_cloud(
        self,
        current_pcd: torch.Tensor,
        goal_pos_xy: torch.Tensor,
        goal_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """生成目标点云（对当前点云施加goal位姿齐次变换）。

        Args:
            current_pcd: 当前点云 [n_envs, n_points, 3]
            goal_pos_xy: 目标位置XY [n_envs, 2]
            goal_yaw: 目标偏航角 [n_envs]

        Returns:
            目标点云 [n_envs, n_points, 3]
        """
        return generate_goal_point_cloud(current_pcd, goal_pos_xy, goal_yaw)

    def compute_icp_error(
        self,
        current_pcd: torch.Tensor,
        goal_pcd: torch.Tensor,
    ) -> torch.Tensor:
        """计算ICP配准误差。

        Args:
            current_pcd: 当前点云 [n_envs, n_points, 3]
            goal_pcd: 目标点云 [n_envs, n_points, 3]

        Returns:
            配准误差 [n_envs]
        """
        _, _, error = icp_align(
            current_pcd, goal_pcd,
            max_iter=self.config.icp_max_iter,
            tol=self.config.icp_tol,
        )
        return error

    def compute_icp_error_downsampled(
        self,
        current_pcd: torch.Tensor,
        goal_pcd: torch.Tensor,
    ) -> torch.Tensor:
        """计算FPS降采样后的ICP配准误差。

        先用FPS将点云降采样到 config.icp_n_points 个点，
        再进行ICP配准计算误差，用于终止条件判定。

        Args:
            current_pcd: 当前点云 [n_envs, n_points, 3]（世界坐标系）
            goal_pcd: 目标点云 [n_envs, n_points, 3]（世界坐标系）

        Returns:
            配准误差 [n_envs]
        """
        n_icp = self.config.icp_n_points
        if n_icp > 0 and n_icp < current_pcd.shape[1]:
            current_fps = farthest_point_sample(current_pcd, n_icp)
            goal_fps = farthest_point_sample(goal_pcd, n_icp)
        else:
            current_fps = current_pcd
            goal_fps = goal_pcd

        _, _, error = icp_align(
            current_fps, goal_fps,
            max_iter=self.config.icp_max_iter,
            tol=self.config.icp_tol,
        )
        return error

    @staticmethod
    def compute_centroid(pcd: torch.Tensor) -> torch.Tensor:
        """计算点云质心。

        Args:
            pcd: 点云 [n_envs, n_points, 3]

        Returns:
            质心 [n_envs, 3]
        """
        return pcd.mean(dim=1)
