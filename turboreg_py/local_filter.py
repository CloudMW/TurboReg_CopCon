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
                 threshold=0.5,
                 k=20):
    corr_kpts_src_sub = corr_kpts_src[cliques_tensor.view(-1)].view(-1, 3, 3)  # [C, 3, 3]
    corr_kpts_dst_sub = corr_kpts_dst[cliques_tensor.view(-1)].view(-1, 3, 3)  # [C, 3, 3]

    # Compute transformation for each clique
    cliques_wise_trans = rigid_transform_3d(corr_kpts_src_sub, corr_kpts_dst_sub)  # [C, 4, 4]

    # Apply each transformation to its corresponding point group
    # Extract rotation and translation from transformation matrices
    cliques_wise_trans_3x3 = cliques_wise_trans[:, :3, :3]  # [C, 3, 3]
    cliques_wise_trans_3x1 = cliques_wise_trans[:, :3, 3:4]  # [C, 3, 1]

    # Transform each point group with its corresponding transformation
    # corr_kpts_src_sub: [C, 3, 3] -> each group has 3 points
    # Apply: R @ points.T + t for each group
    corr_kpts_src_sub_transformed = torch.bmm(cliques_wise_trans_3x3,
                                              corr_kpts_src_sub.permute(0, 2, 1)) + cliques_wise_trans_3x1  # [C, 3, 3]
    corr_kpts_src_sub_transformed = corr_kpts_src_sub_transformed.permute(0, 2, 1)  # [C, 3, 3]

    # Transform source keypoints: R @ kpts_src.T + t
    kpts_src_prime = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
    kpts_src_prime = kpts_src_prime.permute(0, 2, 1)  # [C, M, 3]

    src_knn_points = knn_search(corr_kpts_src_sub_transformed, kpts_src_prime, k=k)  # [C, 3, k, 3]
    dst_knn_points = knn_search(corr_kpts_dst_sub, kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_sub.shape[0], 1, 1), k=k)

    # 计算src和dst对应邻居点之间的最近距离
    neighbor_distances = compute_neighbor_distances(src_knn_points, dst_knn_points).mean(dim=(1, 2)) # [N, 3, k]

    # visualize_knn_neighbors(kpts_src_prime, src_knn_points, corr_kpts_src_sub_transformed)
    # (neighbor_distances)
    ind = (-1*neighbor_distances).topk(k=min(100,neighbor_distances.shape[0]))[1]

    return cliques_tensor[ind]


def knn_search(corr_kpts_src_sub_transformed, kpts_src_prime, k=10):
    # KNN search: For each transformation, find k nearest neighbors for each of the 3 keypoints
    # corr_kpts_src_sub_transformed: [N, 3, 3] - N transformations, 3 keypoints each
    # kpts_src_prime: [N, M, 3] - N transformations, M source points each

    N = corr_kpts_src_sub_transformed.shape[0]
    M = kpts_src_prime.shape[1]

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
    kpts_src_prime_expanded = kpts_src_prime.unsqueeze(1).expand(-1, 3, -1, -1)  # [N, 3, M, 3]

    # Gather k nearest neighbors: [N, 3, k, 3]
    knn_points = torch.gather(kpts_src_prime_expanded, 2, knn_indices_expanded)  # [N, 3, k, 3]
    return knn_points


def compute_neighbor_distances(src_knn_points: torch.Tensor, dst_knn_points: torch.Tensor) -> torch.Tensor:
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

    return min_distances


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
            [0.0, 0.0, 1.0]   # Blue for 3rd keypoint
        ]

        # Create geometries list for this iteration
        geometries = []

        # 收集所有邻居点的索引，用于从源点云中排除
        all_neighbor_indices = set()

        for j in range(3):  # 3 keypoints
            # Create point cloud for neighbors
            neighbors = src_knn_points[i, j, :, :].copy()  # [k, 3]
            print(f"  Keypoint {j+1} ({['Red', 'Green', 'Blue'][j]}): {len(neighbors)} neighbors")
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
            print(f"  Displaying {len(src_only_indices)} source points (excluded {len(all_neighbor_indices)} neighbor points)")
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
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_size * 0.001)
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
