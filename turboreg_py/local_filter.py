import torch
import numpy as np
import open3d as o3d

from turboreg_py.rigid_transform import rigid_transform_3d


def local_filter(cliques_tensor: torch.Tensor,
                 corr_kpts_src: torch.Tensor,
                 corr_kpts_dst: torch.Tensor,
                 kpts_src: torch.Tensor,
                 kpts_dst: torch.Tensor,
                 corr_ind: torch.Tensor,
                 feature_kpts_src: torch.Tensor = None,
                 feature_kpts_dst: torch.Tensor = None,
                 threshold=0.01,
                 k=20,
                 num_cliques=100):
    # Sanitize cliques_tensor early: ensure dtype is long and clamp indices to valid range
    if cliques_tensor.dtype != torch.long:
        cliques_tensor = cliques_tensor.long()

    # num_corr = corr_kpts_src.shape[0]
    # Clamp all indices to [0, num_corr-1] to avoid out-of-bounds indexing
    # cliques_tensor = torch.clamp(cliques_tensor, min=0, max=num_corr - 1)

    k_list = [50,100]
    N, _ = cliques_tensor.shape
    neighbor_distances = torch.Tensor(N, 0).to(corr_kpts_src.device)
    for i in k_list:
        neighbor_distances_one = local_filter_(
            cliques_tensor,
            corr_kpts_src,
            corr_kpts_dst,
            kpts_src,
            kpts_dst,
            corr_ind,
            feature_kpts_src=None,
            feature_kpts_dst=None,
            threshold=threshold,
            k=i,
        )
        neighbor_distances = torch.concat((neighbor_distances, neighbor_distances_one.unsqueeze(-1)), dim=-1)

    neighbor_distances = neighbor_distances.mean(-1)
    big_zero = (neighbor_distances > 0).sum()

    ind = (neighbor_distances).topk(k=min(num_cliques,big_zero.item()))[1]
    return cliques_tensor[ind]


def local_filter_(cliques_tensor: torch.Tensor,
                  corr_kpts_src: torch.Tensor,
                  corr_kpts_dst: torch.Tensor,
                  kpts_src: torch.Tensor,
                  kpts_dst: torch.Tensor,
                  corr_ind: torch.Tensor,
                  feature_kpts_src: torch.Tensor = None,
                  feature_kpts_dst: torch.Tensor = None,
                  threshold=0.5,
                  k=20):
    N, C = cliques_tensor.shape

    # Ensure cliques_tensor is long type and on same device (should already be done in local_filter)
    # if cliques_tensor.dtype != torch.long:
    #     cliques_tensor = cliques_tensor.long()
    # if cliques_tensor.device != corr_kpts_src.device:
    #     cliques_tensor = cliques_tensor.to(corr_kpts_src.device)

    if feature_kpts_dst is None:

        corr_kpts_src_points = corr_kpts_src[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]
        corr_kpts_dst_points = corr_kpts_dst[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]

        # Compute transformation for each clique
        cliques_wise_trans = rigid_transform_3d(corr_kpts_src_points, corr_kpts_dst_points)  # [C, 4, 4]

        # Apply each transformation to its corresponding point group
        # Extract rotation and translation from transformation matrices
        cliques_wise_trans_3x3 = cliques_wise_trans[:, :3, :3]  # [C, 3, 3]
        cliques_wise_trans_3x1 = cliques_wise_trans[:, :3, 3:4]  # [C, 3, 1]

        # Transform each point group with its corresponding transformation
        # corr_kpts_src_sub: [C, 3, 3] -> each group has 3 points
        # Apply: R @ points.T + t for each group
        cliques_src_points_transformed = torch.bmm(cliques_wise_trans_3x3,
                                                  corr_kpts_src_points.permute(0, 2,
                                                                            1)) + cliques_wise_trans_3x1  # [C, 3, 3]
        cliques_src_points_transformed = cliques_src_points_transformed.permute(0, 2, 1)  # [C, 3, 3]

        # Transform source keypoints: R @ kpts_src.T + t
        kpts_src_transformed = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
        kpts_src_transformed = kpts_src_transformed.permute(0, 2, 1)  # [C, M, 3]

        src_knn_points, src_indices = knn_search(cliques_src_points_transformed, kpts_src_transformed, k=k)  # [C, 3, k, 3]
        dst_knn_points, ref_indices = knn_search(corr_kpts_dst_points, kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_points.shape[0], 1, 1), k=k)

        # 计算src和dst对应邻居点之间的最近距离
        # mae = compute_neighbor_distances(src_knn_points, dst_knn_points, threshold).mean(dim=(1, 2))  # [N, 3, k]

        mae_dis_src2dst, mae_dis_dst2_src = compute_neighbor_distances(src_knn_points, dst_knn_points,
                                                                       threshold)  # [N, 3, k]

        src_normals_tensor = compute_normals_o3d(kpts_src, k_neighbors=10)
        dst_normals_tensor = compute_normals_o3d(kpts_dst, k_neighbors=10)
        # Transform source keypoints: R @ kpts_src.T + t
        kpts_src_norm_transformed = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, src_normals_tensor.T)
        kpts_src_norm_transformed = kpts_src_norm_transformed.permute(0, 2, 1)  # [C, M, 3]
        src_knn_corr_norm = kpts_src_norm_transformed[torch.arange(N).view(N, 1, 1), src_indices]
        dst_knn_corr_norm = dst_normals_tensor[ref_indices]
        norm_diff = torch.abs(torch.cosine_similarity(src_knn_corr_norm, dst_knn_corr_norm, dim=-1))


        from turboreg_py.curvature import batch_curvature
        src_curvatures = batch_curvature(kpts_src.unsqueeze(0), k=30)[0]
        dst_curvatures = batch_curvature(kpts_dst.unsqueeze(0), k=30)[0]
        src_knn_corr_curv = src_curvatures[src_indices ]
        dst_knn_corr_curv = dst_curvatures[ref_indices]
        curv_diff = torch.abs(src_knn_corr_curv - dst_knn_corr_curv) / (torch.abs(src_knn_corr_curv) + torch.abs(dst_knn_corr_curv) + 1e-8)

        cliques_knn_overlap = (mae_dis_src2dst<threshold).sum(-1)+ (mae_dis_dst2_src<threshold).sum(-1)
        legal_cliques = ((cliques_knn_overlap>(k/10)).prod(dim=-1, keepdim=False))

        # final_mae = ((mae_dis_src2dst < threshold) * (norm_diff > 0.6) * (curv_diff<0.7)).sum(dim=(1, 2))*legal_cliques

        if legal_cliques.sum() ==0:
            # legal_cliques = ((cliques_knn_overlap > (k / 10)).any(dim=-1, keepdim=False))
            final_mae = (mae_dis_src2dst < threshold).sum(dim=(1, 2))
        else:
            final_mae = (mae_dis_src2dst < threshold).sum(dim=(1, 2)) * legal_cliques
        # visualize_knn_neighbors(kpts_src_transformed, src_knn_points, cliques_src_points_transformed)
        # # Also visualize source+destination together (if destination info available)


        big_zero = (final_mae > 0).sum()
        ind = (final_mae).topk(k=min(20, big_zero.item()))[1]
        vis = False

        if vis:
            try:
                vis_kpts_dst = kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_points.shape[0], 1, 1)
            except Exception:
                vis_kpts_dst = None

            visualize_knn_neighbors_src_dst(
                kpts_src_transformed[ind],
                src_knn_points[ind],
                cliques_src_points_transformed[ind],
                vis_kpts_dst[ind],
                dst_knn_points[ind],
                corr_kpts_dst_points[ind],
                final_mae=(mae_dis_src2dst < threshold).sum(-1)[ind]
            )
    else:

        corr_kpts_src_points = corr_kpts_src[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]
        corr_kpts_dst_points = corr_kpts_dst[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]

        # Compute transformation for each clique
        cliques_wise_trans = rigid_transform_3d(corr_kpts_src_points, corr_kpts_dst_points)  # [C, 4, 4]

        # Apply each transformation to its corresponding point group
        # Extract rotation and translation from transformation matrices
        cliques_wise_trans_3x3 = cliques_wise_trans[:, :3, :3]  # [C, 3, 3]
        cliques_wise_trans_3x1 = cliques_wise_trans[:, :3, 3:4]  # [C, 3, 1]

        # Transform each point group with its corresponding transformation
        # corr_kpts_src_sub: [C, 3, 3] -> each group has 3 points
        # Apply: R @ points.T + t for each group
        cliques_src_points_transformed = torch.bmm(cliques_wise_trans_3x3,
                                                  corr_kpts_src_points.permute(0, 2,
                                                                            1)) + cliques_wise_trans_3x1  # [C, 3, 3]
        cliques_src_points_transformed = cliques_src_points_transformed.permute(0, 2, 1)  # [C, 3, 3]

        # Transform source keypoints: R @ kpts_src.T + t
        kpts_src_transformed = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
        kpts_src_transformed = kpts_src_transformed.permute(0, 2, 1)  # [C, M, 3]

        src_knn_points, knn_indices_src = knn_search(cliques_src_points_transformed, kpts_src_transformed, k=k)  # [C, 3, k, 3]
        dst_knn_points, knn_indices_dst = knn_search(corr_kpts_dst_points,
                                                     kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_points.shape[0], 1, 1),
                                                     k=k)


        # Extract knn features and compute feature correlation
        src_cliques_knn_feature = feature_kpts_src[knn_indices_src]
        dst_cliques_knn_feature = feature_kpts_dst[knn_indices_dst]
        feature_corr = feature_corr_compute(src_cliques_knn_feature, dst_cliques_knn_feature, knn_indices_src, knn_indices_dst,top_k=k*5)
        src_indices = feature_corr[..., 0]
        ref_indices = feature_corr[..., 1]

        src_knn_corr_point =kpts_src_transformed[torch.arange(N).view(N,1,1), src_indices]


        dst_knn_corr_point = kpts_dst[ref_indices]

        mae_dis_src2dst,mae_dis_dst2_src = compute_neighbor_distances(src_knn_corr_point, dst_knn_corr_point, threshold)  # [N, 3, k]

        src_normals_tensor = compute_normals_o3d(kpts_src, k_neighbors=10)
        dst_normals_tensor = compute_normals_o3d(kpts_dst, k_neighbors=10)
        # Transform source keypoints: R @ kpts_src.T + t
        kpts_src_norm_transformed = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, src_normals_tensor.T)
        kpts_src_norm_transformed = kpts_src_norm_transformed.permute(0, 2, 1)  # [C, M, 3]
        src_knn_corr_norm = kpts_src_norm_transformed[torch.arange(N).view(N,1,1), src_indices]
        dst_knn_corr_norm = dst_normals_tensor[ref_indices]
        norm_diff = torch.abs(torch.cosine_similarity(src_knn_corr_norm , dst_knn_corr_norm,dim=-1))

        (mae_dis_src2dst<threshold) + (dst_normals_tensor<threshold)

        final_mae = ((mae_dis_src2dst<threshold )*(norm_diff > 0.4) ).sum(dim=(1,2))

        # visualize_knn_neighbors(kpts_src_transformed, src_knn_points, cliques_src_points_transformed)
        # Also visualize source+destination together (if destination info available)
        # try:
        #     vis_kpts_dst = kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_points.shape[0], 1, 1)
        # except Exception:
        #     vis_kpts_dst = None

        # visualize_knn_neighbors_src_dst(
        #     kpts_src_transformed,
        #     src_knn_points,
        #     cliques_src_points_transformed,
        #     vis_kpts_dst,
        #     dst_knn_points,
        #     corr_kpts_dst_points
        # )


    return final_mae

def feature_corr_src_to_dst(src_knn_points: torch.Tensor, dst_knn_points: torch.Tensor,) -> torch.Tensor:
    """
    计算src和dst对应关键点的邻居点之间的特征相关性（批量化实现）

    参数:
        src_knn_points: 源邻居点特征 [N, 3, k, F]
        dst_knn_points: 目标邻居点特征 [N, 3, k, F]

    返回:
        max_correlations: [N, 3] - 每个变换/关键点的聚合相似度分数（平均top-k相似度）
    """

    # src_knn_points: [N,3,k,F]
    # dst_knn_points: [N,3,k,F]
    assert src_knn_points.ndim == 4 and dst_knn_points.ndim == 4, "Expect input shape [N,3,k,F]"
    N, C, k, F = src_knn_points.shape
    assert dst_knn_points.shape == (N, C, k, F), "src and dst shapes must match"

    device = src_knn_points.device
    eps = 1e-8

    # Merge batch and keypoint dims to process in one batch: B = N * 3
    # B = N * C
    src_flat = src_knn_points  # [B, k, F]
    dst_flat = dst_knn_points  # [B, k, F]

    # Normalize features to unit vectors for normalized L2 (if vector is zero, eps protects)
    src_norm = src_flat / (src_flat.norm(p=2, dim=-1, keepdim=True) + eps)
    dst_norm = dst_flat / (dst_flat.norm(p=2, dim=-1, keepdim=True) + eps)

    # Compute pairwise L2 distances between normalized features: result [B, k, k]
    # Use broadcasting: (src[:, :, None, :] - dst[:, None, :, :])
    diff = src_norm.unsqueeze(-2) - dst_norm.unsqueeze(-3)  # [B, k, k, F]
    dists = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=0.0))  # [B, k, k]

    # Convert distances to similarity scores using exponential kernel (as in forward)
    matching_scores = torch.exp(-dists)  # [B

def feature_corr_compute(src_knn_points: torch.Tensor, dst_knn_points: torch.Tensor,
                         src_indices: torch.Tensor, ref_indices: torch.Tensor,
                         top_k: int = 10, dual_normalization: bool = True) -> torch.Tensor:
    """
    计算src和dst对应关键点的邻居点之间的特征相关性（批量化实现）

    参数:
        src_knn_points: 源邻居点特征 [N, 3, k, F]
        dst_knn_points: 目标邻居点特征 [N, 3, k, F]
        top_k: 每个(源邻居, 目标邻居)对的展平相似度矩阵中选取的top-k数量（用于聚合）
        dual_normalization: 是否使用行/列双向归一化（与原始forward逻辑一致）

    返回:
        max_correlations: [N, 3] - 每个变换/关键点的聚合相似度分数（平均top-k相似度）
    """

    # src_knn_points: [N,3,k,F]
    # dst_knn_points: [N,3,k,F]
    assert src_knn_points.ndim == 4 and dst_knn_points.ndim == 4, "Expect input shape [N,3,k,F]"
    N, C, k, F = src_knn_points.shape
    assert dst_knn_points.shape == (N, C, k, F), "src and dst shapes must match"

    device = src_knn_points.device
    eps = 1e-8

    # Merge batch and keypoint dims to process in one batch: B = N * 3
    # B = N * C
    src_flat = src_knn_points  # [B, k, F]
    dst_flat = dst_knn_points  # [B, k, F]

    # Normalize features to unit vectors for normalized L2 (if vector is zero, eps protects)
    src_norm = src_flat / (src_flat.norm(p=2, dim=-1, keepdim=True) + eps)
    dst_norm = dst_flat / (dst_flat.norm(p=2, dim=-1, keepdim=True) + eps)

    # Compute pairwise L2 distances between normalized features: result [B, k, k]
    # Use broadcasting: (src[:, :, None, :] - dst[:, None, :, :])
    diff = src_norm.unsqueeze(-2) - dst_norm.unsqueeze(-3)  # [B, k, k, F]
    dists = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=0.0))  # [B, k, k]

    # Convert distances to similarity scores using exponential kernel (as in forward)
    matching_scores = torch.exp(-dists)  # [B, k, k]

    if dual_normalization:
        # Row normalization (sum over destination neighbors)
        row_sum = matching_scores.sum(dim=-2, keepdim=True)  # [B, k, 1]
        row_norm = matching_scores / (row_sum + eps)
        # Column normalization (sum over source neighbors)
        col_sum = matching_scores.sum(dim=-3, keepdim=True)  # [B, 1, k]
        col_norm = matching_scores / (col_sum + eps)
        # Geometric-like combination (here multiply) to enhance symmetry
        matching_scores = row_norm * col_norm  # [B, k, k]

    # Flatten per (B) to select top-k correspondences
    flattened = matching_scores.view(N, C, -1)  # [B, k*k]
    K_select = min(top_k, flattened.shape[-1])
    top_vals, top_index = flattened.topk(k=K_select, largest=True)

    # Aggregate top-k values (mean) to produce one score per (B)
    # agg_scores = top_vals.mean(dim=1)  # [B]

    # Reshape back to [N, 3]
    # max_correlations = agg_scores.view(N, C)  # [N, 3]

    # 将1D索引转换回2D索引
    ref_sel_indices = top_index // matching_scores.shape[-1]  # (K,) - 参考超点在有效列表中的索引
    src_sel_indices = top_index % matching_scores.shape[-1]  # (K,) - 源超点在有效列表中的索引
    # ==================== 步骤6: 恢复原始索引 ====================
    # 将有效列表中的索引映射回原始超点列表中的索引
    ref_corr_indices = torch.gather(ref_indices, dim=2, index=ref_sel_indices)  # (K,) - 参考超点在原始列表中的索引
    src_corr_indices = torch.gather(src_indices, dim=2, index=src_sel_indices)  # (K,) - 源超点在原始列表中的索引

    return  torch.concat([src_corr_indices.unsqueeze(-1), ref_corr_indices.unsqueeze(-1)], dim=-1)



def knn_search(corr_kpts_src_sub_transformed, kpts_src_prime, k=10):
    # KNN search: For each transformation, find k nearest neighbors for each of the 3 keypoints
    # corr_kpts_src_sub_transformed: [N, 3, 3] - N transformations, 3 keypoints each
    # kpts_src_prime: [N, M, 3] - N transformations, M source points each

    N = corr_kpts_src_sub_transformed.shape[0]
    M = kpts_src_prime.shape[1]
    C = corr_kpts_src_sub_transformed.shape[1]  # Should be 3
    # Compute pairwise distances for each transformation
    # For each transformation i, compute distance between 3 keypoints and M source points
    # Reshape for broadcasting: [N, 3, 1, 3] - [N, 1, M, 3] = [N, 3, M, 3]
    kpts_expanded = corr_kpts_src_sub_transformed.unsqueeze(2)  # [N, 3, 1, 3]
    src_expanded = kpts_src_prime.unsqueeze(1)  # [N, 1, M, 3]

    # Compute squared Euclidean distances
    dists = torch.sum((kpts_expanded - src_expanded) ** 2, dim=-1)  # [N, 3, M]

    # Find k nearest neighbors for each keypoint
    _, knn_indices = torch.topk(dists, k, dim=2, largest=False)  # [N, 3, k]

    # Gather the k nearest neighbor points
    # Expand indices for gathering: [N, 3, k, 1] -> [N, 3, k, 3]
    knn_indices_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)  # [N, 3, k, 3]

    # Expand kpts_src_prime for gathering: [N, 1, M, 3] -> [N, 3, M, 3]
    kpts_src_prime_expanded = kpts_src_prime.unsqueeze(1).expand(-1, C, -1, -1)  # [N, 3, M, 3]

    # Gather k nearest neighbors: [N, 3, k, 3]
    knn_points = torch.gather(kpts_src_prime_expanded, 2, knn_indices_expanded)  # [N, 3, k, 3]
    return knn_points, knn_indices


def compute_neighbor_distances(src_knn_points: torch.Tensor, dst_knn_points: torch.Tensor,
                               threshold: float) -> torch.Tensor:
    """
    计算src和dst对应关键点的邻居点之间的最近距离

    参数:
        src_knn_points: 源邻居点 [N, 3, k, 3] - N个变换，3个关键点，每个关键点k个邻居
        dst_knn_points: 目标邻居点 [N, 3, k, 3] - N个变换，3个关键点，每个关键点k个邻居

    返回:
        distances: 距离矩阵 [N, 3, k] - 每个源邻居点到dst邻居点的最近距离
    """
    # 使用批量处理，避免循环
    # src_knn_points: [N, 3, k, 3] -> [N, 3, k, 1, 3]
    # dst_knn_points: [N, 3, k, 3] -> [N, 3, 1, k, 3]
    src_expanded = src_knn_points.unsqueeze(3)  # [N, 3, k, 1, 3]
    dst_expanded = dst_knn_points.unsqueeze(2)  # [N, 3, 1, k, 3]

    # 计算所有src和dst邻居点对之间的距离
    # 广播后得到 [N, 3, k, k, 3]
    dist_matrix = torch.sqrt(torch.sum((src_expanded - dst_expanded) ** 2, dim=-1))  # [N, 3, k, k]

    # 对于每个src邻居点，找到最近的dst邻居点的距离
    min_distances, _ = torch.min(dist_matrix, dim=-1)  # [N, 3, k]


    src_expanded_dst2src = src_knn_points.unsqueeze(2)  # [N, 3, k, 1, 3]
    dst_expanded_dst2src = dst_knn_points.unsqueeze(3)  # [N, 3, 1, k, 3]

    # 计算所有src和dst邻居点对之间的距离
    # 广播后得到 [N, 3, k, k, 3]
    dist_matrix_dst2src = torch.sqrt(torch.sum((dst_expanded_dst2src - src_expanded_dst2src) ** 2, dim=-1))  # [N, 3, k, k]

    # 对于每个src邻居点，找到最近的dst邻居点的距离
    min_distances_dst2src, _ = torch.min(dist_matrix_dst2src, dim=-1)  # [N, 3, k]
    # tou = threshold
    # mae = torch.where(min_distances < tou, torch.abs(tou - min_distances) / tou, 0)
    return min_distances, min_distances_dst2src

# def compute_corr_norm_dis (
#         kpts_src: torch.Tensor,
#         kpts_dst: torch.Tensor,
#         threshold: float = 0.5,
#         ) -> torch.Tensor:
#
#     # Get normals for each clique
#     # src_norms = src_normals_tensor[cliques_tensor.view(-1)].view(N, C, 3)  # [N, C, 3]
#     # dst_norms = dst_normals_tensor[cliques_tensor.view(-1)].view(N, C, 3)  # [N, C, 3]


def compute_normals_o3d(points: torch.Tensor, k_neighbors: int = 10) -> torch.Tensor:
    """
    Compute point cloud normals using Open3D (per-point KNN in the same point cloud).
    Returns unit normals on CPU tensor.
    """
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    normals = torch.from_numpy(np.asarray(pcd.normals)).to(points.device)
    # normals = normals / (normals.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return normals.to(points.dtype)


def visualize_knn_neighbors(
        kpts_src: torch.Tensor,
        src_knn_points: torch.Tensor,
        corr_kpts_src_sub_transformed: torch.Tensor = None,
        point_size: float = 2.0,
        keypoint_size: float = 20,
        neighbor_size: float = 20.0
):
    """
    可视化每个变换的源点云和KNN邻居点

    参数:
        kpts_src: 源点云关键点 [N, M, 3] - N个变换后的源点云
        src_knn_points: KNN邻居点 [N, 3, k, 3] - N个变换，每个有3个关键点，每个关键点有k个邻居
        corr_kpts_src_sub_transformed: 变换后的对应关键点 [N, 3, 3] (可选)
        point_size: 源点云点的大小
        keypoint_size: 关键��的大小（用于球体半径，单位与点云坐标一致）
        neighbor_size: 邻居点的大小
    """
    # Convert to numpy
    if isinstance(kpts_src, torch.Tensor):
        kpts_src = kpts_src.cpu().numpy()
    if isinstance(src_knn_points, torch.Tensor):
        src_knn_points = src_knn_points.cpu().numpy()
    if corr_kpts_src_sub_transformed is not None and isinstance(corr_kpts_src_sub_transformed, torch.Tensor):
        corr_kpts_src_sub_transformed = corr_kpts_src_sub_transformed.cpu().numpy()

    N = src_knn_points.shape[0]  # Number of transformations

    # Iterate through each transformation
    for i in range(N):
        print(f"\n=== Visualization for Transformation {i + 1}/{N} ===")
        print(f"Source points: {kpts_src[i].shape}")
        print(f"KNN points shape: {src_knn_points[i].shape}")

        # Visualize the 3 keypoints and their neighbors
        colors = [
            [1.0, 0.0, 0.0],  # Red for 1st keypoint
            [0.0, 1.0, 0.0],  # Green for 2nd keypoint
            [0.0, 0.0, 1.0]  # Blue for 3rd keypoint
        ]

        # Create geometries list for this iteration
        geometries = []

        # 收集所有邻居点的索引，用于从源点云中排除
        all_neighbor_indices = set()

        for j in range(3):  # 3 keypoints
            # Create point cloud for neighbors
            neighbors = src_knn_points[i, j, :, :].copy()  # [k, 3]
            print(f"  Keypoint {j + 1} ({['Red', 'Green', 'Blue'][j]}): {len(neighbors)} neighbors")
            print(f"    Neighbor positions range: min={neighbors.min(axis=0)}, max={neighbors.max(axis=0)}")

            # 找到这些邻居点在源点云中的索引
            # 通过计算距离找到完全匹配的点
            for neighbor in neighbors:
                distances = np.linalg.norm(kpts_src[i] - neighbor, axis=1)
                matching_indices = np.where(distances < 1e-6)[0]
                all_neighbor_indices.update(matching_indices.tolist())

        print(f"  Total neighbor point indices to exclude: {len(all_neighbor_indices)}")

        # 创建过滤后的源点云（去除邻居点）
        all_indices = set(range(len(kpts_src[i])))
        src_only_indices = list(all_indices - all_neighbor_indices)

        if len(src_only_indices) > 0:
            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(kpts_src[i][src_only_indices].copy())
            pcd_src.colors = o3d.utility.Vector3dVector(np.tile([0.7, 0.7, 0.7], (len(src_only_indices), 1)))
            geometries.append(pcd_src)
            print(
                f"  Displaying {len(src_only_indices)} source points (excluded {len(all_neighbor_indices)} neighbor points)")
        else:
            print(f"  No source points to display (all are neighbors)")

        for j in range(3):  # 3 keypoints
            # Create point cloud for neighbors
            neighbors = src_knn_points[i, j, :, :].copy()  # [k, 3]

            pcd_neighbors = o3d.geometry.PointCloud()
            pcd_neighbors.points = o3d.utility.Vector3dVector(neighbors)
            # Set each point's color individually
            neighbor_colors = np.array([colors[j] for _ in range(len(neighbors))])
            pcd_neighbors.colors = o3d.utility.Vector3dVector(neighbor_colors)
            geometries.append(pcd_neighbors)

            print(f"    Added {len(neighbors)} points with color {colors[j]}")

            # If transformed keypoints are provided, visualize them as spheres
            if corr_kpts_src_sub_transformed is not None:
                keypoint = corr_kpts_src_sub_transformed[i, j, :].copy()  # [3]
                print(f"    Keypoint location: {keypoint}")

                # Create a sphere for the keypoint at origin
                sphere = o3d.geometry.TriangleMesh.create_sphere()
                sphere.scale(keypoint_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                sphere.compute_vertex_normals()
                # Set vertex colors
                vertices = np.asarray(sphere.vertices)
                num_vertices = len(vertices)
                vertex_colors = np.array([colors[j] for _ in range(num_vertices)])
                sphere.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                # Transform the sphere to the keypoint location
                transform_matrix = np.eye(4)
                transform_matrix[:3, 3] = keypoint
                sphere.transform(transform_matrix)
                geometries.append(sphere)

        print(f"Total geometries to render: {len(geometries)}")

        # Use draw_geometries for simpler rendering
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Transformation {i + 1}/{N}",
            width=1024,
            height=768,
            point_show_normal=False
        )


def visualize_knn_neighbors_src_dst(
        kpts_src: torch.Tensor,
        src_knn_points: torch.Tensor,
        corr_kpts_src_sub_transformed: torch.Tensor = None,
        kpts_dst: torch.Tensor = None,
        dst_knn_points: torch.Tensor = None,
        corr_kpts_dst_sub_transformed: torch.Tensor = None,
        point_size: float = 2.0,
        keypoint_size: float = 20.0,
        neighbor_size: float = 6.0,
        final_mae=None
):
    """
    可视化源和目标点云以及它们的关键点和邻居点（批量，单窗口，按键切换）

    - 键盘控制：N 下一帧，P 上一帧，Q 退出
    - 使用一个窗口，避免多窗口/阻塞造成的卡死
    - 主点云通过 RenderOption.point_size 控制大小；关键点/邻居用小球体（可单独控制尺寸）
    """
    # Convert tensors to numpy
    if isinstance(kpts_src, torch.Tensor):
        kpts_src = kpts_src.cpu().numpy()
    if isinstance(src_knn_points, torch.Tensor):
        src_knn_points = src_knn_points.cpu().numpy()
    if corr_kpts_src_sub_transformed is not None and isinstance(corr_kpts_src_sub_transformed, torch.Tensor):
        corr_kpts_src_sub_transformed = corr_kpts_src_sub_transformed.cpu().numpy()

    if kpts_dst is not None and isinstance(kpts_dst, torch.Tensor):
        kpts_dst = kpts_dst.cpu().numpy()
    if dst_knn_points is not None and isinstance(dst_knn_points, torch.Tensor):
        dst_knn_points = dst_knn_points.cpu().numpy()
    if corr_kpts_dst_sub_transformed is not None and isinstance(corr_kpts_dst_sub_transformed, torch.Tensor):
        corr_kpts_dst_sub_transformed = corr_kpts_dst_sub_transformed.cpu().numpy()

    N = src_knn_points.shape[0]

    # Colors for source keypoints (RGB) and destination keypoints (magenta/cyan/yellow)
    src_colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    dst_colors = [np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])]

    def build_geometries(i: int):
        if (final_mae is not None):
            print(f"\n=== Visualization for Transformation {i + 1}/{N} | Final MAE: {final_mae[i]} ===")

        geometries = []

        # 收集源中需排除的邻居点（避免重叠）
        all_neighbor_indices = set()
        for j in range(3):
            neighbors = src_knn_points[i, j, :, :].copy()
            for neighbor in neighbors:
                distances = np.linalg.norm(kpts_src[i] - neighbor, axis=1)
                matching = np.where(distances < 1e-6)[0]
                all_neighbor_indices.update(matching.tolist())

        # 源点云（去重叠）
        all_idx = set(range(len(kpts_src[i])))
        src_only_idx = list(all_idx - all_neighbor_indices)
        if len(src_only_idx) > 0:
            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(kpts_src[i][src_only_idx].copy())
            colors_src = np.tile([0.7, 0.7, 0.7], (len(src_only_idx), 1)).astype(np.float64)
            pcd_src.colors = o3d.utility.Vector3dVector(colors_src)
            geometries.append(pcd_src)

        # 目标点云
        if kpts_dst is not None and len(kpts_dst) > i:
            pcd_dst = o3d.geometry.PointCloud()
            pcd_dst.points = o3d.utility.Vector3dVector(kpts_dst[i].copy())
            colors_dst = np.tile([0.5, 0.5, 0.9], (len(kpts_dst[i]), 1)).astype(np.float64)
            pcd_dst.colors = o3d.utility.Vector3dVector(colors_dst)
            geometries.append(pcd_dst)

        # 源邻居与关键点（球体）
        for j in range(3):
            neighbors = src_knn_points[i, j, :, :].copy()
            color = src_colors[j]
            for pt in neighbors:
                sphere = o3d.geometry.TriangleMesh.create_sphere()
                sphere.scale(neighbor_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color(color.tolist())
                T = np.eye(4);
                T[:3, 3] = pt
                sphere.transform(T)
                geometries.append(sphere)

            if corr_kpts_src_sub_transformed is not None:
                kp = corr_kpts_src_sub_transformed[i, j, :].copy()
                sphere_k = o3d.geometry.TriangleMesh.create_sphere()
                sphere_k.scale(keypoint_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                sphere_k.compute_vertex_normals()
                sphere_k.paint_uniform_color(color.tolist())
                T = np.eye(4);
                T[:3, 3] = kp
                sphere_k.transform(T)
                geometries.append(sphere_k)

        # 目标邻居与关键点（球体）
        if dst_knn_points is not None:
            for j in range(3):
                neighbors = dst_knn_points[i, j, :, :].copy()
                color = dst_colors[j]
                for pt in neighbors:
                    sphere = o3d.geometry.TriangleMesh.create_sphere()
                    sphere.scale(neighbor_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                    sphere.compute_vertex_normals()
                    sphere.paint_uniform_color(color.tolist())
                    T = np.eye(4);
                    T[:3, 3] = pt
                    sphere.transform(T)
                    geometries.append(sphere)

                if corr_kpts_dst_sub_transformed is not None:
                    kp = corr_kpts_dst_sub_transformed[i, j, :].copy()
                    sphere_k = o3d.geometry.TriangleMesh.create_sphere()
                    sphere_k.scale(keypoint_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                    sphere_k.compute_vertex_normals()
                    sphere_k.paint_uniform_color(color.tolist())
                    T = np.eye(4);
                    T[:3, 3] = kp
                    sphere_k.transform(T)
                    geometries.append(sphere_k)

        return geometries

    # 渲染选项设置
    def set_render_opts(vis: o3d.visualization.Visualizer):
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        try:
            setattr(opt, 'light_on', False)
        except Exception:
            pass
        try:
            setattr(opt, 'mesh_show_back_face', True)
        except Exception:
            pass

    # 状态与回调
    state = {'idx': 0}

    def show_current(vis):
        vis.clear_geometries()

        geoms = build_geometries(state['idx'])
        for g in geoms:
            vis.add_geometry(g)
        set_render_opts(vis)
        vis.update_renderer()
        return False

    def cb_next(vis):
        if state['idx'] < N - 1:
            state['idx'] += 1
            print(f"Switch to {state['idx'] + 1}/{N}")
            return show_current(vis)
        return False

    def cb_prev(vis):
        if state['idx'] > 0:
            state['idx'] -= 1
            print(f"Switch to {state['idx'] + 1}/{N}")
            return show_current(vis)
        return False

    def cb_quit(vis):
        vis.close()
        return False

    # 初始几何体 + 键盘回调
    init_geoms = build_geometries(0)
    key_callbacks = {
        ord('N'): cb_next,
        ord('P'): cb_prev,
        ord('Q'): cb_quit,
    }
    print("Controls: N(next), P(prev), Q(quit)")

    o3d.visualization.draw_geometries_with_key_callbacks(
        init_geoms,
        key_callbacks,
        window_name="KNN neighbors (src+dst)",
        width=1024,
        height=768
    )
