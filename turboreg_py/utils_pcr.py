"""
Point Cloud Registration Utilities
Contains helper functions for point cloud processing
"""

import torch
from torch.cuda import device

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None  # type: ignore
    print("Warning: Open3D not found. Falling back to PyTorch k-NN normal estimation.")


def select_non_coplanar_cliques_ranked(cliques_points_src, M, threshold=1e-3):
    """
    从N个团中挑选M个不共面的团，按"不共面程度"排序
    （最小奇异值越大，表示越不共面）

    Args:
        cliques_points_src: [N, C, 3] tensor
        M: 要挑选的团数量
        threshold: 共面判断阈值

    Returns:
        selected_indices: list，挑选出的团的索引
        selected_cliques: [M, C, 3] 挑选出的团
        singular_values: 对应的最小奇异值
    """
    N, C, _ = cliques_points_src.shape

    if M > N:
        raise ValueError(f"M ({M}) 不能大于团的总数 ({N})")

    # 计算质心
    centroids = cliques_points_src.mean(dim=1, keepdim=True)  # [N, 1, 3]

    # 中心化
    centered = cliques_points_src - centroids  # [N, C, 3]

    # 批量SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # 获取每个团的最小奇异值
    min_singular_values = S[:, -1]  # [N]

    # 找到不共面的团（最小奇异值 >= threshold）
    non_coplanar_mask = min_singular_values >= threshold
    non_coplanar_indices = torch.where(non_coplanar_mask)[0]
    non_coplanar_values = min_singular_values[non_coplanar_indices]

    num_non_coplanar = len(non_coplanar_indices)

    if num_non_coplanar < M:
        print(f"警告：只有 {num_non_coplanar} 个不共面的团，少于要求的 {M} 个")
        print(f"将返回所有 {num_non_coplanar} 个不共面的团")
        # 按最小奇异值降序排序
        sorted_indices = torch.argsort(non_coplanar_values, descending=True)
        selected_local_indices = sorted_indices
    else:
        # 选择最小奇异值最大的M个团（最不共面的）
        sorted_indices = torch.argsort(non_coplanar_values, descending=True)
        selected_local_indices = sorted_indices[:M]

    selected_indices = non_coplanar_indices[selected_local_indices].tolist()
    selected_singular_values = non_coplanar_values[selected_local_indices]
    selected_cliques = cliques_points_src[selected_indices]

    return selected_indices, selected_cliques, selected_singular_values.tolist()


def coplanar_constraint_more_points(
        cliques_tensor: torch.Tensor,
        corr_kpts_src: torch.Tensor,
        corr_kpts_dst: torch.Tensor,
        kpts_src: torch.Tensor,
        kpts_dst: torch.Tensor,
        corr_ind: torch.Tensor,
        plus_threshold: float = 0,
        k=100
) -> torch.Tensor:
    N,C = cliques_tensor.shape

    cliques_points_src = corr_kpts_src[cliques_tensor.view(-1)].view(-1,C, 3)
    cliques_points_dst = corr_kpts_dst[cliques_tensor.view(-1)].view(-1, C, 3)
    selected_index  ,_,_ = select_non_coplanar_cliques_ranked(cliques_points_src, k, threshold=1e-3)
    cliques_tensor = cliques_tensor[selected_index]
    return  cliques_tensor

def coplanar_constraint(
        cliques_tensor: torch.Tensor,
        corr_kpts_src: torch.Tensor,
        corr_kpts_dst: torch.Tensor,
        kpts_src: torch.Tensor,
        kpts_dst: torch.Tensor,
        corr_ind: torch.Tensor,
        plus_threshold: float = 0,
        k=100
) -> torch.Tensor:
    """
    Filter cliques based on coplanar constraint using normal vectors

    Args:
        cliques_tensor: Clique indices [N, 3]
        corr_kpts_src: Source keypoints [M, 3]
        corr_kpts_dst: Target keypoints [M, 3]
        threshold: Threshold for normal similarity (default: 0.5)

    Returns:
        Filtered cliques tensor
    """
    N ,C= cliques_tensor.shape
    original_device = corr_kpts_src.device

    # Move to CPU for Open3D; fallback to PyTorch if Open3D not available
    # kpts_src_cpu = kpts_src.cpu()
    # kpts_dst_cpu = kpts_dst.cpu()
    # pts_src_cpu = pts_src.cpu()
    # pts_dst_cpu = pts_dst.cpu()
    # cliques_cpu = cliques_tensor.cpu()

    src_normals_tensor = compute_normals_o3d(kpts_src, k_neighbors=10)[corr_ind[:,0]]
    dst_normals_tensor = compute_normals_o3d(kpts_dst, k_neighbors=10)[corr_ind[:,1]]
    src_norms_all_i = src_normals_tensor.unsqueeze(1)
    src_norms_all_j = src_normals_tensor.unsqueeze(0)
    src_dot_all_mean = torch.abs((src_norms_all_i * src_norms_all_j).sum(dim=-1)).mean()  # [N, C, C]

    dst_norms_all_i = dst_normals_tensor.unsqueeze(1)
    dst_norms_all_j = dst_normals_tensor.unsqueeze(0)
    dst_dot_all_mean = torch.abs((dst_norms_all_i * dst_norms_all_j).sum(dim=-1)).mean()  # [N, C, C]

    # Get normals for each clique
    src_norms = src_normals_tensor[cliques_tensor.view(-1)].view(N, C, 3)  # [N, C, 3]
    dst_norms = dst_normals_tensor[cliques_tensor.view(-1)].view(N, C, 3)  # [N, C, 3]

    # Compute pairwise cosine similarities for all C points
    # src_norms: [N, C, 3], expand to [N, C, 1, 3] and [N, 1, C, 3]
    src_norms_i = src_norms.unsqueeze(2)  # [N, C, 1, 3]
    src_norms_j = src_norms.unsqueeze(1)  # [N, 1, C, 3]

    # Compute dot product: [N, C, C]
    src_dot = (src_norms_i * src_norms_j).sum(dim=-1)  # [N, C, C]

    # Same for destination
    dst_norms_i = dst_norms.unsqueeze(2)  # [N, C, 1, 3]
    dst_norms_j = dst_norms.unsqueeze(1)  # [N, 1, C, 3]
    dst_dot = (dst_norms_i * dst_norms_j).sum(dim=-1)  # [N, C, C]

    # Take absolute value of cosine similarities
    src_sim_matrix = torch.abs(src_dot)  # [N, C, C]
    dst_sim_matrix = torch.abs(dst_dot)  # [N, C, C]

    # Extract upper triangular values (excluding diagonal) for each clique
    # For C points, we have C*(C-1)/2 unique pairs
    triu_indices = torch.triu_indices(C, C, offset=1, device=src_norms.device)
    src_sims = src_sim_matrix[:, triu_indices[0], triu_indices[1]]  # [N, C*(C-1)/2]
    dst_sims = dst_sim_matrix[:, triu_indices[0], triu_indices[1]]  # [N, C*(C-1)/2]

    # Concatenate all pairwise similarities
    all_sims = torch.cat([src_sims, dst_sims], dim=-1)  # [N, C*(C-1)]

    # Keep cliques with minimum similarity below threshold (match C++)
    # min_sims = all_sims.mean(dim=-1)

    # score,ind = (-1*min_sims).topk(k=400)
    threshold  = (dst_dot_all_mean+src_dot_all_mean)/2 + plus_threshold
    min_sims = torch.where(all_sims < threshold,1,0).sum(-1)
    big_zero = (min_sims>0).sum()
    score, ind = min_sims.topk(k=min(k,big_zero))
    # mask = min_sims < threshold
    # filtered_cliques = cliques_tensor[mask]
    filtered_cliques = cliques_tensor[ind]
    if original_device.type in ['cuda', 'xpu']:
        filtered_cliques = filtered_cliques.to(original_device)

    return filtered_cliques


def compute_normals_o3d(points: torch.Tensor, k_neighbors: int = 10) -> torch.Tensor:
    """
    Compute point cloud normals using Open3D (per-point KNN in the same point cloud).
    Returns unit normals on CPU tensor.
    """
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    normals = torch.from_numpy(np.asarray(pcd.normals) ).to(points.device)
    # normals = normals / (normals.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return normals



# Import numpy if open3d is available
if HAS_OPEN3D:
    import numpy as np
