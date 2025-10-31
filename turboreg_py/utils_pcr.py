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


def coplanar_constraint(
        cliques_tensor: torch.Tensor,
        corr_kpts_src: torch.Tensor,
        corr_kpts_dst: torch.Tensor,
        kpts_src: torch.Tensor,
        kpts_dst: torch.Tensor,
        corr_ind: torch.Tensor,
        threshold: float = 0.5
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
    min_sims = all_sims.mean(dim=-1)

    score,ind = (-1*min_sims).topk(k=400)

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
