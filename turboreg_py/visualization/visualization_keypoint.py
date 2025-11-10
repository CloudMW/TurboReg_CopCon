import numpy as np
import torch
import open3d as o3d


import numpy as np
import torch
import open3d as o3d

def visualize_keypoint(src: torch.Tensor,
                       dst: torch.Tensor,
                       keypoint_index_src: torch.Tensor,
                       keypoint_index_dst: torch.Tensor,
                       gt_trans: torch.Tensor):
    """
    可视化关键点，背景为黑色：
    - 对 src 应用 gt_trans 后显示（浅黄色），src 关键点为深黄色
    - 显示 dst（浅蓝色），dst 关键点为深蓝色
    输入：src/dst (N,3) tensors，keypoint_index_* 为一维索引张量，gt_trans 为 (4,4)
    """
    assert src.dim() == 2 and src.size(1) == 3
    assert dst.dim() == 2 and dst.size(1) == 3
    assert gt_trans.shape == (4, 4)

    src_np = src.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()
    kp_src_idx = keypoint_index_src.long().cpu().numpy() if keypoint_index_src is not None else np.array([], dtype=int)
    kp_dst_idx = keypoint_index_dst.long().cpu().numpy() if keypoint_index_dst is not None else np.array([], dtype=int)
    gt_np = gt_trans.detach().cpu().numpy()

    # 对 src 应用齐次变换
    ones = np.ones((src_np.shape[0], 1), dtype=src_np.dtype)
    src_h = np.concatenate([src_np, ones], axis=1)
    src_warp = (gt_np @ src_h.T).T[:, :3]

    # 颜色定义
    light_yellow = np.array([1.0, 1.0, 0.8])
    dark_yellow  = np.array([1.0, 0.8, 0.0])
    light_blue   = np.array([0.7, 0.85, 1.0])
    dark_blue    = np.array([0.0, 0.2, 0.8])

    # 构建颜色数组并防越界
    src_colors = np.tile(light_yellow, (src_warp.shape[0], 1))
    if kp_src_idx.size:
        kp_src_idx = np.clip(kp_src_idx, 0, src_warp.shape[0] - 1)
        src_colors[kp_src_idx] = dark_yellow

    dst_colors = np.tile(light_blue, (dst_np.shape[0], 1))
    if kp_dst_idx.size:
        kp_dst_idx = np.clip(kp_dst_idx, 0, dst_np.shape[0] - 1)
        dst_colors[kp_dst_idx] = dark_blue

    # 创建点云对象
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_warp)
    pcd_src.colors = o3d.utility.Vector3dVector(src_colors)

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst_np)
    pcd_dst.colors = o3d.utility.Vector3dVector(dst_colors)

    # 关键点球体（可选）
    spheres = []
    # 根据两个点云的范围估计球体半径
    combined = np.vstack([src_warp, dst_np]) if src_warp.size and dst_np.size else src_warp if src_warp.size else dst_np
    bbox_scale = np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)) if combined.size else 1.0
    sphere_radius = max(0.002, bbox_scale * 0.005)

    for idx in np.unique(kp_src_idx):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=float(sphere_radius))
        s.translate(src_warp[int(idx)])
        s.paint_uniform_color(dark_yellow.tolist())
        s.compute_vertex_normals()
        spheres.append(s)
    for idx in np.unique(kp_dst_idx):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=float(sphere_radius))
        s.translate(dst_np[int(idx)])
        s.paint_uniform_color(dark_blue.tolist())
        s.compute_vertex_normals()
        spheres.append(s)

    # 使用 Visualizer 设置黑色背景并显示
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Keypoint Visualization", width=1280, height=720)
    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])  # 黑色背景
    vis.add_geometry(pcd_src)
    vis.add_geometry(pcd_dst)
    for g in spheres:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()

