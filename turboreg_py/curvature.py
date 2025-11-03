import torch
def batch_curvature(points: torch.Tensor, k: int = 30) -> torch.Tensor:
    """
    批量计算点云曲率 (B, N, 3)
    返回 (B, N)
    """
    B, N, _ = points.shape

    # Step 1: 计算两两距离
    dist = torch.cdist(points, points)  # [B, N, N]
    knn_idx = dist.topk(k, largest=False).indices  # [B, N, k]

    # Step 2: 构造邻域点集
    idx_base = torch.arange(B, device=points.device)[:, None, None]
    neighbors = points[idx_base, knn_idx, :]  # [B, N, k, 3]

    # Step 3: 去中心化
    centroid = neighbors.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
    centered = neighbors - centroid  # [B, N, k, 3]

    # Step 4: 协方差矩阵 (batch)
    # cov = (X^T X) / k
    cov = torch.matmul(centered.transpose(-2, -1), centered) / k  # [B, N, 3, 3]

    # Step 5: 求特征值
    eigvals = torch.linalg.eigvalsh(cov)  # [B, N, 3]
    eigvals, _ = torch.sort(eigvals, dim=-1)

    # Step 6: 曲率计算
    curvature = eigvals[..., 0] / (eigvals.sum(dim=-1) + 1e-8)  # [B, N]

    return curvature