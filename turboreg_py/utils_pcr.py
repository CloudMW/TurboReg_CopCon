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
        kpts_src: torch.Tensor,
        kpts_dst: torch.Tensor,
        pts_src: torch.Tensor,
        pts_dst: torch.Tensor,
        corr_ind: torch.Tensor,
        threshold: float = 0.5
) -> torch.Tensor:
    """
    Filter cliques based on coplanar constraint using normal vectors

    Args:
        cliques_tensor: Clique indices [N, 3]
        kpts_src: Source keypoints [M, 3]
        kpts_dst: Target keypoints [M, 3]
        threshold: Threshold for normal similarity (default: 0.5)

    Returns:
        Filtered cliques tensor
    """
    N = cliques_tensor.size(0)
    original_device = kpts_src.device

    # Move to CPU for Open3D; fallback to PyTorch if Open3D not available
    # kpts_src_cpu = kpts_src.cpu()
    # kpts_dst_cpu = kpts_dst.cpu()
    # pts_src_cpu = pts_src.cpu()
    # pts_dst_cpu = pts_dst.cpu()
    # cliques_cpu = cliques_tensor.cpu()

    src_normals_tensor = compute_normals_o3d(pts_src, k_neighbors=10)[corr_ind[:,0]]
    dst_normals_tensor = compute_normals_o3d(pts_dst, k_neighbors=10)[corr_ind[:,1]]


    # Get normals for each clique
    src_norms = src_normals_tensor[cliques_tensor.view(-1)].view(N, 3, 3)  # [N, 3, 3]
    dst_norms = dst_normals_tensor[cliques_tensor.view(-1)].view(N, 3, 3)  # [N, 3, 3]

    # Extract normals for three points
    src_n0, src_n1, src_n2 = src_norms[:, 0], src_norms[:, 1], src_norms[:, 2]
    dst_n0, dst_n1, dst_n2 = dst_norms[:, 0], dst_norms[:, 1], dst_norms[:, 2]

    # Cosine similarity absolute value
    src_sim01 = torch.abs(torch.cosine_similarity(src_n0, src_n1))  # [N]
    src_sim02 = torch.abs(torch.cosine_similarity(src_n0, src_n2))
    src_sim12 = torch.abs(torch.cosine_similarity(src_n1, src_n2))

    dst_sim01 = torch.abs(torch.cosine_similarity(dst_n0, dst_n1))
    dst_sim02 = torch.abs(torch.cosine_similarity(dst_n0, dst_n2))
    dst_sim12 = torch.abs(torch.cosine_similarity(dst_n1, dst_n2))

    all_sims = torch.stack([
        src_sim01, src_sim02, src_sim12,
        dst_sim01, dst_sim02, dst_sim12
    ], dim=-1)  # [N, 6]

    # Keep cliques with minimum similarity below threshold (match C++)
    min_sims = all_sims.mean(dim=-1)
    mask = min_sims < threshold
    filtered_cliques = cliques_tensor[mask]

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
