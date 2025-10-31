import torch
import numpy as np
from scipy.optimize import minimize

from turboreg_py.rigid_transform import rigid_transform_3d

def fit_minimum_enclosing_sphere(points):
    """
    @param points: Nx3 array-like or torch.Tensor of 3D points
    @return: center (3,), radius (float)
    计算包含所有点的最小外接球
    使用优化算法求解最小外接球问题
    返回球心坐标和半径
    """
    points = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
    n_points = points.shape[0]

    if n_points == 1:
        return points[0], 0.0
    elif n_points == 2:
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[1] - points[0]) / 2
        return center, radius

    def objective_function(params, points):
        center = np.array(params)
        distances = np.linalg.norm(points - center, axis=1)
        # 目标：最小化最大距离（外接球半径）
        return np.max(distances)

    # 多个初始猜测以找到全局最优解
    best_center = None
    best_radius = float('inf')

    # 尝试不同的初始点
    initial_guesses = [
        np.mean(points, axis=0),  # 质心
        np.median(points, axis=0),  # 中位数点
        points[np.random.randint(n_points)],  # 随机点
    ]

    # 添加边界点作为初始猜测
    for dim in range(3):
        min_idx = np.argmin(points[:, dim])
        max_idx = np.argmax(points[:, dim])
        initial_guesses.append(points[min_idx])
        initial_guesses.append(points[max_idx])

    for initial_guess in initial_guesses:
        try:
            result = minimize(objective_function, initial_guess, args=(points,),
                            method='L-BFGS-B',
                            options={'ftol': 1e-9, 'gtol': 1e-9})

            if result.success:
                center = result.x
                distances = np.linalg.norm(points - center, axis=1)
                radius = np.max(distances)

                if radius < best_radius:
                    best_center = center
                    best_radius = radius
        except:
            continue

    # 如果优化失败，使用简单的质心方法
    if best_center is None:
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.max(distances)
        return center, radius

    return best_center, best_radius

def extract_points_in_sphere(points, center, radius, tolerance=1e-6):
    """
    提取被球体包围的点

    Args:
        points: 点云 (torch.Tensor 或 numpy.array)
        center: 球心坐标 (torch.Tensor 或 numpy.array)
        radius: 球体半径 (float 或 torch.Tensor scalar)
        tolerance: 容差，用于处理浮点数精度问题

    Returns:
        included_points: 被球体包围的点 (torch.Tensor)
        included_indices: 被包围点的原始索引 (torch.Tensor)
    """
    # 判断输入类型
    is_tensor = isinstance(points, torch.Tensor)

    # 将 center 转换为 numpy
    if isinstance(center, torch.Tensor):
        center_np = center.cpu().numpy()
    else:
        center_np = center

    # 将 radius 转换为 float
    if isinstance(radius, torch.Tensor):
        radius_val = radius.item()
    else:
        radius_val = radius

    # 转换 points 为 numpy 进行计算
    if is_tensor:
        points_np = points.cpu().numpy()
    else:
        points_np = points

    # 计算每个点到球心的距离
    distances = np.linalg.norm(points_np - center_np, axis=1)

    # 找到在球体内部或边界上的点（考虑容差）
    included_mask = distances <= (radius_val + tolerance)
    included_indices = np.where(included_mask)[0]

    if is_tensor:
        # 返回torch tensor格式
        included_points = points[included_indices]
        included_indices = torch.from_numpy(included_indices).long()
    else:
        included_points = points_np[included_indices]
        included_indices = included_indices

    return included_points, included_indices

def sphere_filter(cliques_tensor: torch.Tensor,
                 corr_kpts_src: torch.Tensor,
                 corr_kpts_dst: torch.Tensor,
                 kpts_src: torch.Tensor,
                 kpts_dst: torch.Tensor,
                 corr_ind: torch.Tensor,
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
    corr_kpts_src_sub_transformed = corr_kpts_src_sub_transformed.permute(0, 2, 1)  # [N, C, 3]

    # Transform source keypoints: R @ kpts_src.T + t
    kpts_src_prime = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
    kpts_src_prime = kpts_src_prime.permute(0, 2, 1)  # [C, M, 3]
    mae = torch.tensor([],dtype=torch.float32).to(cliques_tensor.device)
    for i in range(N):
        clique_tensor = cliques_tensor[i]
        # cli_src_points = corr_kpts_src_sub_transformed[i][corr_ind[clique_tensor,0]]
        # cli_dst_points = corr_kpts_dst[corr_ind[clique_tensor,1]]

        sphere_coeff_src_cli = fit_minimum_enclosing_sphere( corr_kpts_src_sub_transformed[i])
        sphere_coeff_dst_cli = fit_minimum_enclosing_sphere(corr_kpts_dst_sub[i])
        sphere_point_src,_ = extract_points_in_sphere(kpts_src_prime[i], sphere_coeff_src_cli[0], sphere_coeff_src_cli[1])
        sphere_point_dst,_ = extract_points_in_sphere(kpts_dst, sphere_coeff_dst_cli[0], sphere_coeff_dst_cli[1])

        dist_matrix = torch.sqrt(torch.sum((sphere_point_src.unsqueeze(1)- sphere_point_dst.unsqueeze(0)) ** 2, dim=-1))  # [N, 3, k, k]

        # 对于每个src邻居点，找到最近的dst邻居点的距离
        min_distances, _ = torch.min(dist_matrix, dim=-1)  # [N, 3, k]
        tou = 0.05
        mae = torch.concat((mae,torch.where(min_distances < tou, torch.abs(tou - min_distances) / tou,0).mean().unsqueeze(0)),0)

        # visualize_sphere_points(
        #     kpts_src_prime[i],
        #     kpts_dst,
        #     sphere_point_src=sphere_point_src,
        #     sphere_point_dst=sphere_point_dst,point_size=5)
    ind = (mae).topk(k=min(20,N))[1]
    return cliques_tensor[ind]
def visualize_sphere_points(
    kpts_src_prime_i,
    kpts_dst,
    sphere_point_src=None,
    sphere_point_dst=None,
    point_size: float = 5.0,
    window_name: str = "Sphere Filter Visualization"
):
    """
    显示原点云 kpts_src_prime[i] 与目标点云 kpts_dst，以及原点云中的 sphere_point_src 和目标点云中的 sphere_point_dst。
    - 原/目标点云使用浅色
    - sphere_point_src / sphere_point_dst 使用深色

    参数:
        kpts_src_prime_i: [M1,3] 源点云（第 i 个变换后的）(torch.Tensor 或 numpy.ndarray)
        kpts_dst: [M2,3] 目标点云 (torch.Tensor 或 numpy.ndarray)
        sphere_point_src: [m1,3] 源球内点 (可选)
        sphere_point_dst: [m2,3] 目标球内点 (可选)
        point_size: 渲染点大小
        window_name: 窗口标题
    """
    # 延迟导入，避免修改全局依赖
    import numpy as _np
    import torch as _torch
    try:
        import open3d as _o3d
    except Exception as e:
        raise RuntimeError("需要 open3d 以进行可视化，请先安装: pip install open3d") from e

    def _to_numpy(arr):
        if arr is None:
            return None
        if isinstance(arr, _torch.Tensor):
            return arr.detach().cpu().numpy()
        return _np.asarray(arr)

    src = _to_numpy(kpts_src_prime_i)
    dst = _to_numpy(kpts_dst)
    src_s = _to_numpy(sphere_point_src)
    dst_s = _to_numpy(sphere_point_dst)

    # 在显示前，剔除主点云中与球内点重叠的部分
    def _remove_overlap(main_arr, sub_arr):
        if main_arr is None or sub_arr is None or main_arr.size == 0 or sub_arr.size == 0:
            return main_arr
        # 使用哈希集合进行高效去重
        sub_set = set(map(tuple, sub_arr))
        main_filtered = [p for p in main_arr if tuple(p) not in sub_set]
        return _np.array(main_filtered) if main_filtered else _np.empty((0, 3), dtype=main_arr.dtype)

    src_main_no_overlap = _remove_overlap(src, src_s)
    dst_main_no_overlap = _remove_overlap(dst, dst_s)

    geoms = []

    # 颜色：浅色用于主云，深色用于球内点
    color_src_main = _np.array([0.80, 0.80, 0.80], dtype=_np.float64)  # 浅灰
    color_dst_main = _np.array([0.60, 0.70, 1.00], dtype=_np.float64)  # 浅蓝
    color_src_sph  = _np.array([1.00, 0.10, 0.10], dtype=_np.float64)  # 深红
    color_dst_sph  = _np.array([0.10, 1.00, 0.00], dtype=_np.float64)  # 深蓝

    # 源主云（已去重）
    if src_main_no_overlap is not None and src_main_no_overlap.size > 0:
        pcd_src = _o3d.geometry.PointCloud()
        pcd_src.points = _o3d.utility.Vector3dVector(src_main_no_overlap.astype(_np.float64, copy=False))
        pcd_src.colors = _o3d.utility.Vector3dVector(_np.tile(color_src_main, (src_main_no_overlap.shape[0], 1)))
        geoms.append(pcd_src)

    # 目标主云（已去重）
    if dst_main_no_overlap is not None and dst_main_no_overlap.size > 0:
        pcd_dst = _o3d.geometry.PointCloud()
        pcd_dst.points = _o3d.utility.Vector3dVector(dst_main_no_overlap.astype(_np.float64, copy=False))
        pcd_dst.colors = _o3d.utility.Vector3dVector(_np.tile(color_dst_main, (dst_main_no_overlap.shape[0], 1)))
        geoms.append(pcd_dst)

    # 源球内点（深色）
    if src_s is not None and src_s.size > 0:
        pcd_src_s = _o3d.geometry.PointCloud()
        pcd_src_s.points = _o3d.utility.Vector3dVector(src_s.astype(_np.float64, copy=False))
        pcd_src_s.colors = _o3d.utility.Vector3dVector(_np.tile(color_src_sph, (src_s.shape[0], 1)))
        geoms.append(pcd_src_s)

    # 目标球内点（深色）
    if dst_s is not None and dst_s.size > 0:
        pcd_dst_s = _o3d.geometry.PointCloud()
        pcd_dst_s.points = _o3d.utility.Vector3dVector(dst_s.astype(_np.float64, copy=False))
        pcd_dst_s.colors = _o3d.utility.Vector3dVector(_np.tile(color_dst_sph, (dst_s.shape[0], 1)))
        geoms.append(pcd_dst_s)

    # 使用 Visualizer 设置点大小
    vis = _o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = _np.asarray([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()
