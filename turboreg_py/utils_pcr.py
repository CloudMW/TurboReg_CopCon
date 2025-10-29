"""
Point Cloud Registration Utilities
Contains helper functions for point cloud processing
"""

import torch
from typing import Optional

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not found. coplanar_constraint will use simplified implementation.")


def coplanar_constraint(
    cliques_tensor: torch.Tensor,
    kpts_src: torch.Tensor,
    kpts_dst: torch.Tensor,
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
    M = kpts_src.size(0)
    
    # Save original device
    original_device = kpts_src.device
    
    # Move to CPU (Open3D only works on CPU)
    kpts_src_cpu = kpts_src.cpu()
    kpts_dst_cpu = kpts_dst.cpu()
    cliques_cpu = cliques_tensor.cpu()
    
    if HAS_OPEN3D:
        # Use Open3D to compute normals
        src_normals_tensor = compute_normals_o3d(kpts_src_cpu)
        dst_normals_tensor = compute_normals_o3d(kpts_dst_cpu)
    else:
        # Use simplified normal estimation (cross product method)
        src_normals_tensor = compute_normals_simple(kpts_src_cpu, cliques_cpu)
        dst_normals_tensor = compute_normals_simple(kpts_dst_cpu, cliques_cpu)
        
        # If simplified method, directly return filtered cliques
        return filter_by_normals_simple(
            cliques_cpu, src_normals_tensor, dst_normals_tensor, threshold, original_device
        )
    
    # Get normals for each clique
    src_norms = src_normals_tensor[cliques_cpu.view(-1)].view(N, 3, 3)  # [N, 3, 3]
    dst_norms = dst_normals_tensor[cliques_cpu.view(-1)].view(N, 3, 3)  # [N, 3, 3]
    
    # Extract normals for three points
    src_n0 = src_norms[:, 0]  # [N, 3]
    src_n1 = src_norms[:, 1]
    src_n2 = src_norms[:, 2]
    
    dst_n0 = dst_norms[:, 0]
    dst_n1 = dst_norms[:, 1]
    dst_n2 = dst_norms[:, 2]
    
    # Compute normal similarities (cosine similarity absolute value)
    src_sim01 = torch.abs((src_n0 * src_n1).sum(-1))  # [N]
    src_sim02 = torch.abs((src_n0 * src_n2).sum(-1))
    src_sim12 = torch.abs((src_n1 * src_n2).sum(-1))
    
    dst_sim01 = torch.abs((dst_n0 * dst_n1).sum(-1))
    dst_sim02 = torch.abs((dst_n0 * dst_n2).sum(-1))
    dst_sim12 = torch.abs((dst_n1 * dst_n2).sum(-1))
    
    # Stack all similarities
    all_sims = torch.stack([
        src_sim01, src_sim02, src_sim12,
        dst_sim01, dst_sim02, dst_sim12
    ], dim=-1)  # [N, 6]
    
    # Get minimum similarity for each clique
    min_sims = all_sims.min(dim=-1)[0]  # [N]
    
    # Filter: keep cliques with min similarity < threshold
    mask = min_sims < threshold
    filtered_cliques = cliques_cpu[mask]
    
    # Move back to original device
    if original_device.type in ['cuda', 'xpu']:
        filtered_cliques = filtered_cliques.to(original_device)
    
    return filtered_cliques


def compute_normals_o3d(points: torch.Tensor, k_neighbors: int = 10) -> torch.Tensor:
    """
    Compute point cloud normals using Open3D
    
    Args:
        points: Point cloud [N, 3]
        k_neighbors: Number of neighbors for normal estimation
    
    Returns:
        Normal vectors [N, 3]
    """
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )
    
    # Convert back to tensor
    normals = torch.from_numpy(np.asarray(pcd.normals)).float()
    
    return normals


def compute_normals_simple(
    points: torch.Tensor,
    cliques: torch.Tensor
) -> torch.Tensor:
    """
    Simplified normal computation using cross product of edges
    This is a fallback when Open3D is not available
    
    Args:
        points: Point cloud [N, 3]
        cliques: Clique indices [C, 3]
    
    Returns:
        Pseudo-normals based on triangles [C, 3]
    """
    # For each clique, compute normal from the triangle
    p0 = points[cliques[:, 0]]  # [C, 3]
    p1 = points[cliques[:, 1]]
    p2 = points[cliques[:, 2]]
    
    # Compute edges
    v1 = p1 - p0
    v2 = p2 - p0
    
    # Cross product to get normal
    normals = torch.cross(v1, v2, dim=-1)  # [C, 3]
    
    # Normalize
    normals = normals / (torch.norm(normals, p=2, dim=-1, keepdim=True) + 1e-8)
    
    return normals


def filter_by_normals_simple(
    cliques: torch.Tensor,
    src_normals: torch.Tensor,
    dst_normals: torch.Tensor,
    threshold: float,
    original_device: torch.device
) -> torch.Tensor:
    """
    Filter cliques using simplified normal comparison
    
    Args:
        cliques: Clique indices [N, 3]
        src_normals: Source normals [N, 3]
        dst_normals: Target normals [N, 3]
        threshold: Similarity threshold
        original_device: Device to return result on
    
    Returns:
        Filtered cliques
    """
    # Compute normal similarity between src and dst for each clique
    normal_sim = torch.abs((src_normals * dst_normals).sum(-1))  # [N]
    
    # Keep cliques with high normal agreement (similar orientation)
    # and low similarity (not coplanar, which means threshold should filter coplanar ones)
    mask = normal_sim > (1.0 - threshold)  # High similarity means same orientation
    
    filtered_cliques = cliques[mask]
    
    # Move back to original device
    if original_device.type in ['cuda', 'xpu']:
        filtered_cliques = filtered_cliques.to(original_device)
    
    return filtered_cliques


# Import numpy if open3d is available
if HAS_OPEN3D:
    import numpy as np

