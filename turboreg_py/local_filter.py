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
                 threshold=0.01,
                 k=20,
                 num_cliques = 100):
    k_list = [25,50,100]
    N,_ = cliques_tensor.shape
    neighbor_distances = torch.Tensor(N,0).to(corr_kpts_src.device)
    for i in k_list:
        neighbor_distances_one = local_filter_(
            cliques_tensor,
            corr_kpts_src,
            corr_kpts_dst,
            kpts_src,
            kpts_dst,
            corr_ind,
            threshold=threshold,
            k=i,
        )
        neighbor_distances=torch.concat((neighbor_distances, neighbor_distances_one.unsqueeze(-1)), dim=-1)

    neighbor_distances = neighbor_distances.mean(-1)
    ind = (neighbor_distances).topk(k=min(num_cliques,neighbor_distances.shape[0]))[1]
    return cliques_tensor[ind]

def local_filter_(cliques_tensor: torch.Tensor,
                 corr_kpts_src: torch.Tensor,
                 corr_kpts_dst: torch.Tensor,
                 kpts_src: torch.Tensor,
                 kpts_dst: torch.Tensor,
                 corr_ind: torch.Tensor,
                 threshold=0.5,
                 k=20):
    N,C= cliques_tensor.shape
    corr_kpts_src_sub = corr_kpts_src[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]
    corr_kpts_dst_sub = corr_kpts_dst[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]

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
    mae = compute_neighbor_distances(src_knn_points, dst_knn_points,threshold).mean(dim=(1, 2)) # [N, 3, k]

    # visualize_knn_neighbors(kpts_src_prime, src_knn_points, corr_kpts_src_sub_transformed)
    # Also visualize source+destination together (if destination info available)
    # try:
    #     vis_kpts_dst = kpts_dst.unsqueeze(0).repeat(corr_kpts_dst_sub.shape[0], 1, 1)
    # except Exception:
    #     vis_kpts_dst = None
    #
    # visualize_knn_neighbors_src_dst(
    #     kpts_src_prime,
    #     src_knn_points,
    #     corr_kpts_src_sub_transformed,
    #     vis_kpts_dst,
    #     dst_knn_points,
    #     corr_kpts_dst_sub
    # )


    return mae


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
    return knn_points


def compute_neighbor_distances(src_knn_points: torch.Tensor, dst_knn_points: torch.Tensor,threshold:float) -> torch.Tensor:
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
    tou = threshold
    mae =torch.where( min_distances<tou,torch.abs(tou - min_distances)/tou,0)
    return mae


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
        neighbor_size: float = 6.0
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
                T = np.eye(4); T[:3, 3] = pt
                sphere.transform(T)
                geometries.append(sphere)

            if corr_kpts_src_sub_transformed is not None:
                kp = corr_kpts_src_sub_transformed[i, j, :].copy()
                sphere_k = o3d.geometry.TriangleMesh.create_sphere()
                sphere_k.scale(keypoint_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                sphere_k.compute_vertex_normals()
                sphere_k.paint_uniform_color(color.tolist())
                T = np.eye(4); T[:3, 3] = kp
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
                    T = np.eye(4); T[:3, 3] = pt
                    sphere.transform(T)
                    geometries.append(sphere)

                if corr_kpts_dst_sub_transformed is not None:
                    kp = corr_kpts_dst_sub_transformed[i, j, :].copy()
                    sphere_k = o3d.geometry.TriangleMesh.create_sphere()
                    sphere_k.scale(keypoint_size * 0.001, center=np.array([0.0, 0.0, 0.0]))
                    sphere_k.compute_vertex_normals()
                    sphere_k.paint_uniform_color(color.tolist())
                    T = np.eye(4); T[:3, 3] = kp
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
