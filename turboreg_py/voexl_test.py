import numpy
import numpy as np
import open3d as o3d


import torch


def mls_upsample_pointcloud(
        pc: torch.Tensor,
        scale: float = 2.0,
        M: int = None,
        k: int = 10,
        sigma: float = 0.1
) -> torch.Tensor:
    """
    移动最小二乘法（MLS）点云上采样（PyTorch GPU加速版）
    Args:
        pc: 原始点云，形状 [N, 3]（N为点数，3为x/y/z三维坐标）
        scale: 上采样倍率（默认2倍，M = N×scale）
        M: 目标点数（指定后scale失效，需M > N）
        k: 每个候选点的局部邻域点数（默认10，需k ≤ N）
        sigma: 高斯权重的标准差（控制权重衰减速度，需根据点云分辨率调整）
    Returns:
        pc_upsampled: 上采样后点云，形状 [M, 3]
    """
    N = pc.shape[0]
    assert pc.shape[1] == 3, "点云必须是 [N,3] 三维坐标"
    assert k <= N, f"邻域点数k={k}不能大于原始点数N={N}"

    # 1. 确定目标点数M
    if M is None:
        M = int(N * scale)
    assert M > N, "目标点数M必须大于原始点数N"

    # 2. 生成候选采样点（原始点云包围盒内均匀采样）
    min_coords = pc.min(dim=0).values  # [3]：x_min/y_min/z_min
    max_coords = pc.max(dim=0).values  # [3]：x_max/y_max/z_max
    # 生成M个均匀分布的候选点（[0,1]→映射到包围盒）
    candidate_points = torch.rand(M, 3, dtype=pc.dtype, device=pc.device)
    candidate_points = min_coords + (max_coords - min_coords) * candidate_points  # [M,3]

    # 3. 为每个候选点找原始点云的k个最近邻（用torch.cdist高效计算）
    # 距离矩阵：[M, N]（候选点i到原始点j的欧氏距离）
    dist_matrix = torch.cdist(candidate_points, pc, p=2.0)
    # 取每个候选点的k个最近邻（距离最小的k个）：返回(距离, 索引)，形状[M,k]
    k_nn_dists, k_nn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
    # 提取每个候选点的k个邻域点：[M, k, 3]
    k_nn_points = pc[k_nn_indices]  # 高级索引：pc[N,3] → [M,k,3]

    # 4. MLS核心：加权拟合局部曲面并投影候选点
    # 4.1 计算高斯权重（距离越近，权重越大）：[M, k]
    weights = torch.exp(-k_nn_dists ** 2 / (2 * sigma ** 2))  # 高斯核：w = exp(-d²/(2σ²))
    weights = weights.unsqueeze(-1)  # [M, k, 1]（适配广播）

    # 4.2 局部坐标系构建（简化版：用邻域点中心归一化，避免数值不稳定）
    # 计算每个候选点邻域的中心点：[M, 3]
    nn_centers = k_nn_points.mean(dim=1, keepdim=True)  # [M,1,3]
    # 邻域点相对于中心点的偏移：[M, k, 3]
    nn_offsets = k_nn_points - nn_centers
    # 候选点相对于中心点的偏移：[M, 1, 3]
    candidate_offset = candidate_points.unsqueeze(1) - nn_centers

    # 4.3 加权最小二乘拟合线性曲面（简化为一阶多项式，平衡效率和精度）
    # 构建设计矩阵A：[M, k, 4]（每个邻域点的基函数：1, x, y, z）
    A = torch.cat([
        torch.ones_like(nn_offsets[..., :1]),  # 常数项 [M,k,1]
        nn_offsets  # 线性项 [M,k,3]
    ], dim=-1)  # [M, k, 4]

    # 加权设计矩阵：A_weighted = A * weights（每个元素乘权重）
    A_weighted = A * weights  # [M, k, 4]

    # 构建目标向量b：邻域点的原始坐标（此处拟合自身，即b=nn_offsets）
    b = nn_offsets  # [M, k, 3]
    b_weighted = b * weights  # [M, k, 3]

    # 求解最小二乘：A^T * W * A * coeff = A^T * W * b → coeff = (A^T W A)⁻¹ A^T W b
    # 计算A^T W A：[M, 4, 4]（每个候选点的4×4矩阵）
    ATA = torch.matmul(A_weighted.transpose(1, 2), A)  # [M,4,k] @ [M,k,4] → [M,4,4]
    # 计算A^T W b：[M, 4, 3]（每个候选点的4×3矩阵）
    ATb = torch.matmul(A_weighted.transpose(1, 2), b_weighted)  # [M,4,k] @ [M,k,3] → [M,4,3]

    # 求逆（用伪逆避免奇异矩阵）：[M,4,4]
    ATA_inv = torch.linalg.pinv(ATA)  # PyTorch 1.9+ 支持，更稳定

    # 求解系数coeff：[M,4,3]
    coeff = torch.matmul(ATA_inv, ATb)  # [M,4,4] @ [M,4,3] → [M,4,3]

    # 4.4 候选点投影到拟合曲面（用系数计算投影后的坐标）
    # 构建候选点的设计矩阵：[M,1,4]（1, x_candidate, y_candidate, z_candidate）
    candidate_A = torch.cat([
        torch.ones_like(candidate_offset[..., :1]),  # [M,1,1]
        candidate_offset  # [M,1,3]
    ], dim=-1)  # [M,1,4]

    # 投影后的偏移：[M,1,3]
    projected_offset = torch.matmul(candidate_A, coeff)  # [M,1,4] @ [M,4,3] → [M,1,3]

    # 还原到全局坐标系：[M,3]
    pc_upsampled = nn_centers.squeeze(1) + projected_offset.squeeze(1)

    return pc_upsampled
def visualize_point_cloud_with_voxel_grid_lines(points, voxel_size=0.02):
    def nn_upsample_pointcloud(
            pc: torch.Tensor,
            scale: float = 2.0,
            M: int = None
    ) -> torch.Tensor:
        """
        3D 点云最近邻插值上采样（PyTorch 高效版）
        Args:
            pc: 原始点云，形状 [N, 3]（N 为点数，3 为 x,y,z 三维坐标）
            scale: 上采样倍率（默认 2 倍，即 M = N × scale）
            M: 目标点数（若指定，scale 失效；需满足 M > N）
        Returns:
            pc_upsampled: 上采样后点云，形状 [M, 3]
        """
        N = pc.shape[0]
        assert pc.shape[1] == 3, "点云必须是 [N,3] 三维坐标"

        # 1. 确定目标点数 M
        if M is None:
            M = int(N * scale)
        assert M > N, "目标点数 M 必须大于原始点数 N"

        # 2. 在原始点云的包围盒内均匀生成 M 个目标采样点
        # 计算原始点云的空间包围盒（x/y/z 最小/最大值）
        min_coords = pc.min(dim=0).values  # [3]：x_min, y_min, z_min
        max_coords = pc.max(dim=0).values  # [3]：x_max, y_max, z_max

        # 生成 M 个均匀分布的目标点（[0,1] 归一化后映射到包围盒）
        target_points = torch.rand(M, 3, dtype=pc.dtype, device=pc.device)  # [M,3] ∈ [0,1]
        target_points = min_coords + (max_coords - min_coords) * target_points  # 映射到原始点云空间

        # 3. 为每个目标点找原始点云中的最近邻（用 torch.cdist 高效计算距离矩阵）
        # 计算目标点与原始点的欧氏距离矩阵：[M, N]（dist[i,j] = 目标点 i 到原始点 j 的距离）
        dist_matrix = torch.cdist(target_points, pc, p=2.0)

        # 找到每个目标点的最近邻索引（每行最小距离对应的原始点索引）
        nn_indices = torch.argmin(dist_matrix, dim=1)  # [M]：每个目标点的最近原始点索引

        # 4. 最近邻插值：目标点坐标 = 最近原始点坐标
        pc_upsampled = pc[nn_indices]  # [M,3]：高级索引批量赋值

        return pc_upsampled




    # o3d.visualization.draw_geometries([upsample_pcd])

    points_numpy = points.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_numpy)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([downsampled_pcd])
    print(f"downsampled_pcd size :{len(downsampled_pcd.points)}")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(downsampled_pcd,
                                                                voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])


