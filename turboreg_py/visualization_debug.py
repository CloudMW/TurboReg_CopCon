import numpy as np
import open3d as o3d
def transform_points(points, transform_matrix):
    """

    将点云坐标系为A的点云通过转换矩阵转换为坐标系为B的点云
    :param points: 点云坐标系为A的点云，形状为(n, 4)，其中n为点的个数，每个点为(x, y, z, 1)
    :param transform_matrix: 转换矩阵，形状为(4, 4)
    :return: 坐标系为B的点云，形状为(n, 4)
    """
    # 添加最后一列，将点云表示为齐次坐标形式
    points_homogeneous = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    # 转换点云坐标
    transformed_points_homogeneous = np.dot(points_homogeneous, transform_matrix.T)
    """
    # 也可以python中使用@操作：points_homogeneous @ transform_matrix.T
    # 或者也可以写成下面形式
    # transformed_points_homogeneous = np.dot(transform_matrix.T, points_homogeneous).T
    # 可以将RT矩阵拆开成R、t；可写成
    # transformed_points_homogeneous = np.dot(points_homogeneous, R) + t   # 推荐
    # 或者transformed_points_homogeneous = np.dot(R, points_homogeneous.T).T + t  # 会有误差
    # 
    """
    # 归一化处理，将齐次坐标转换为三维坐标
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3:]
    return transformed_points

def visualization_from_clique_configurable(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    src_kpts: np.ndarray,
    tgt_kpts: np.ndarray,
    corr_idx: np.ndarray,
    GTmat: np.ndarray,
    best_trans: np.ndarray,
    best_clique: set,
    estimate_normals: bool = False,
    line_width: float = 50.0,
    point_size: float = 1.0,
    show_all_correspondences: bool = True,
    gt_label: np.ndarray = None
):
    """
    可视化基于最大团的点云配准结果（可配置版本）。

    参数:
        src_points (np.ndarray): 源点云，形状为 (N, 3)
        tgt_points (np.ndarray): 目标点云，形状为 (M, 3)
        src_kpts (np.ndarray): 源点云关键点，形状为 (K, 3)
        tgt_kpts (np.ndarray): 目标点云关键点，形状为 (K, 3)
        corr_idx (np.ndarray): 匹配索引，形状为 (K, 2)
        GTmat (np.ndarray): 真值变换矩阵，形状为 (4, 4)
        best_trans (np.ndarray): 最优变换矩阵，形状为 (4, 4)
        best_clique (set): 最优团（最大团）索引集合
        estimate_normals (bool): 是否计算点云法线，默认 False
        line_width (float): 线段宽度（像素），默认 20.0
        point_size (float): 点的大小（像素），默认 1.0
        show_all_correspondences (bool): 是否显示所有对应点连线，默认 True
        gt_label (np.ndarray): 对应点标签，形状为 (K,)，1 为正确对应（绿色），0 为错误对应（红色），默认 None
    """
    # 提取最佳团的对应点
    src_best_pts = src_kpts[np.array(list(best_clique))]
    tgt_best_pts = tgt_kpts[corr_idx[:,1]][np.array(list(best_clique))]

    src_gt_pts = src_best_pts.copy()
    src_best_pts = transform_points(src_best_pts, best_trans)
    src_gt_pts = transform_points(src_gt_pts, GTmat)

    # 构建最佳团的 LineSet 对象
    line_points = [point for point in np.concatenate((src_best_pts, tgt_best_pts), axis=0)]
    line_points_gt = [point for point in np.concatenate((src_gt_pts, tgt_best_pts), axis=0)]

    n = int(len(line_points)/2)
    front_indices = np.array([[i, i + 1] for i in range(n - 1)] + [[n-1, 0]])
    back_indices = np.array([[i, i + 1] for i in range(n, 2 * n - 1)] + [[2*n-1, n]])
    lines = np.vstack([front_indices, back_indices])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set_gt = o3d.geometry.LineSet()
    line_set_gt.points = o3d.utility.Vector3dVector(line_points_gt)
    line_set_gt.lines = o3d.utility.Vector2iVector(lines)

    # 设置线段颜色
    colors = np.array([[1, 0, 0] for _ in range(len(front_indices))] +
                      [[0, 1, 0] for _ in range(len(back_indices))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set_gt.colors = o3d.utility.Vector3dVector(colors)

    # 创建点云对象
    src_ply_best_trans = o3d.geometry.PointCloud()
    src_ply_best_trans.points = o3d.utility.Vector3dVector(src_points)
    src_ply_gt_trans = o3d.geometry.PointCloud()
    src_ply_gt_trans.points = o3d.utility.Vector3dVector(src_points)
    tgt_ply = o3d.geometry.PointCloud()
    tgt_ply.points = o3d.utility.Vector3dVector(tgt_points)

    src_ply_best_trans.paint_uniform_color([1, 0.706, 0])
    src_ply_gt_trans.paint_uniform_color([1, 0.706, 0])
    tgt_ply.paint_uniform_color([0, 0.651, 0.929])
    src_ply_best_trans.transform(best_trans)
    src_ply_gt_trans.transform(GTmat)

    # 关键点点云
    tgt_kpts_pcd = o3d.geometry.PointCloud()
    tgt_kpts_pcd.points = o3d.utility.Vector3dVector(tgt_kpts)
    src_kpts_pred_trans = o3d.geometry.PointCloud()
    src_kpts_pred_trans.points = o3d.utility.Vector3dVector(src_kpts)
    src_kpts_gt_trans = o3d.geometry.PointCloud()
    src_kpts_gt_trans.points = o3d.utility.Vector3dVector(src_kpts)

    src_kpts_pred_trans.paint_uniform_color([1, 0.706, 0])
    src_kpts_gt_trans.paint_uniform_color([1, 0.706, 0])
    tgt_kpts_pcd.paint_uniform_color([0, 0.651, 0.929])

    src_kpts_gt_trans.transform(GTmat)
    src_kpts_pred_trans.transform(best_trans)

    # 创建所有对应点的连线
    corr_lines_gt = None
    corr_lines_pred = None
    if show_all_correspondences:
        # GT 变换后的对应连线
        src_kpts_gt_transformed = transform_points(src_kpts, GTmat)
        tgt_kpts_corr = tgt_kpts[corr_idx[:, 1]]

        # 构建连线点和索引
        corr_points_gt = np.vstack([src_kpts_gt_transformed, tgt_kpts_corr])
        corr_lines_idx = np.array([[i, i + len(src_kpts)] for i in range(len(src_kpts))])

        corr_lines_gt = o3d.geometry.LineSet()
        corr_lines_gt.points = o3d.utility.Vector3dVector(corr_points_gt)
        corr_lines_gt.lines = o3d.utility.Vector2iVector(corr_lines_idx)

        # 根据 gt_label 设置颜色
        if gt_label is not None:
            # gt_label 为 1 的设置为绿色，为 0 的设置为红色
            corr_colors_gt = np.array([[0, 1, 0] if gt_label[i] == 1 else [1, 0, 0]
                                       for i in range(len(corr_lines_idx))])
        else:
            # 默认蓝色连线
            corr_colors_gt = np.array([[0, 0, 1] for _ in range(len(corr_lines_idx))])

        corr_lines_gt.colors = o3d.utility.Vector3dVector(corr_colors_gt)

        # Pred 变换后的对应连线
        src_kpts_pred_transformed = transform_points(src_kpts, best_trans)
        corr_points_pred = np.vstack([src_kpts_pred_transformed, tgt_kpts_corr])

        corr_lines_pred = o3d.geometry.LineSet()
        corr_lines_pred.points = o3d.utility.Vector3dVector(corr_points_pred)
        corr_lines_pred.lines = o3d.utility.Vector2iVector(corr_lines_idx)

        # 根据 gt_label 设置颜色
        if gt_label is not None:
            # gt_label 为 1 的设置为绿色，为 0 的设置为红色
            corr_colors_pred = np.array([[0, 1, 0] if gt_label[i] == 1 else [1, 0, 0]
                                         for i in range(len(corr_lines_idx))])
        else:
            # 默认青色连线
            corr_colors_pred = np.array([[0, 1, 1] for _ in range(len(corr_lines_idx))])

        corr_lines_pred.colors = o3d.utility.Vector3dVector(corr_colors_pred)

    # 根据参数决定是否计算法线
    if estimate_normals:
        if not tgt_ply.has_normals():
            tgt_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if not src_ply_best_trans.has_normals():
            src_ply_best_trans.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if not src_ply_gt_trans.has_normals():
            src_ply_gt_trans.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 创建可视化窗口
    vis1 = o3d.visualization.VisualizerWithKeyCallback()
    vis2 = o3d.visualization.VisualizerWithKeyCallback()
    vis3 = o3d.visualization.VisualizerWithKeyCallback()
    vis4 = o3d.visualization.VisualizerWithKeyCallback()
    vis5 = o3d.visualization.VisualizerWithKeyCallback()


    vis5.create_window(window_name="Ground Truth (Full)", width=960, height=500, left=0, top=0)
    vis5.add_geometry(tgt_ply)
    vis5.add_geometry(src_ply_gt_trans)
    # vis1.add_geometry(line_set_gt)


    vis1.create_window(window_name="Ground Truth (Full) with line", width=960, height=500, left=0, top=0)
    vis1.add_geometry(tgt_ply)
    vis1.add_geometry(src_ply_gt_trans)
    vis1.add_geometry(line_set_gt)

    vis2.create_window(window_name="Prediction (Full)", width=900, height=500, left=1000, top=0)
    vis2.add_geometry(tgt_ply)
    vis2.add_geometry(src_ply_best_trans)
    vis2.add_geometry(line_set)

    vis3.create_window(window_name="Ground Truth (Keypoints)", width=900, height=500, left=0, top=600)
    vis3.add_geometry(tgt_kpts_pcd)
    vis3.add_geometry(src_kpts_gt_trans)
    vis3.add_geometry(line_set_gt)
    if show_all_correspondences and corr_lines_gt is not None:
        vis3.add_geometry(corr_lines_gt)

    vis4.create_window(window_name="Prediction (Keypoints)", width=900, height=500, left=1000, top=600)
    vis4.add_geometry(tgt_kpts_pcd)
    vis4.add_geometry(src_kpts_pred_trans)
    vis4.add_geometry(line_set)
    if show_all_correspondences and corr_lines_pred is not None:
        vis4.add_geometry(corr_lines_pred)

    # 设置渲染选项
    for vis in [vis1, vis2, vis3, vis4,vis5]:
        render_option = vis.get_render_option()
        render_option.line_width = line_width
        render_option.point_size = point_size

    ctr1 = vis1.get_view_control()
    ctr2 = vis2.get_view_control()
    ctr3 = vis3.get_view_control()
    ctr4 = vis4.get_view_control()
    ctr5 = vis5.get_view_control()
    # 注册键盘事件来关闭窗口
    def close_all_windows(vis):
        vis1.destroy_window()
        vis2.destroy_window()
        vis3.destroy_window()
        vis4.destroy_window()
        vis5.destroy_window()
        return True

    vis1.register_key_callback(ord("Q"), close_all_windows)
    vis2.register_key_callback(ord("Q"), close_all_windows)
    vis3.register_key_callback(ord("Q"), close_all_windows)
    vis4.register_key_callback(ord("Q"), close_all_windows)
    vis5.register_key_callback(ord("Q"), close_all_windows)
    # 同步所有窗口的视角
    while True:
        param = ctr1.convert_to_pinhole_camera_parameters()
        ctr2.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        ctr3.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        ctr4.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        ctr5.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        vis1.poll_events()
        vis1.update_renderer()
        vis2.poll_events()
        vis2.update_renderer()
        vis3.poll_events()
        vis3.update_renderer()
        vis4.poll_events()
        vis4.update_renderer()
        vis5.poll_events()
        vis5.update_renderer()

        # 检查是否有窗口被关闭
        if not vis1.poll_events() or not vis2.poll_events() or not vis3.poll_events() or not vis4.poll_events() or not vis5.poll_events():
            break

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()
    vis5.destroy_window()
