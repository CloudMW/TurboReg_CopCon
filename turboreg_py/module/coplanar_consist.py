import torch
import open3d as o3d
import numpy as np
def coplanar_consists(src_point:torch.Tensor ,tgt_point:torch.Tensor,corr_index:torch.Tensor):

    src_numpy = src_point.cpu().numpy()

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_numpy)
    tgt_numpy = tgt_point.cpu().numpy()
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_numpy)

    src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    src_pcd_normal_numpy = np.asarray(src_pcd.normals)
    tgt_pcd_normal_numpy = np.asarray(tgt_pcd.normals)

    src_pcd_normal_tensor = torch.from_numpy(src_pcd_normal_numpy).to(src_point.device)
    tgt_pcd_normal_tensor = torch.from_numpy(tgt_pcd_normal_numpy).to(tgt_point.device)

    src_corr_normal = src_pcd_normal_tensor[corr_index[:,0]]
    src_CopCons =  torch.abs(torch.matmul(src_corr_normal, src_corr_normal.t()))
    tgt_corr_normal = tgt_pcd_normal_tensor[corr_index[:,1]]
    tgt_CopCons = torch.abs(torch.matmul(tgt_corr_normal, tgt_corr_normal.t()))
    ## 过滤掉差异过大的点

    normal_diff = torch.abs(src_CopCons - tgt_CopCons)
    normal_diff_file = torch.where(normal_diff<0.3, torch.ones_like(normal_diff), torch.zeros_like(normal_diff))

    ## 过滤掉共面点
    normal_copcons = torch.where((src_CopCons <0.5)&( tgt_CopCons <0.5), torch.ones_like(src_CopCons), torch.zeros_like(src_CopCons))

    return normal_copcons

    # src_normal_dot_tgt_normal = torch.einsum('cn,cn->c', src_pcd_normal_tensor, tgt_pcd_normal_tensor)
    # src_normal_dot_tgt_normal = src_normal_dot_tgt_normal.view(-1,1)
    #
    # src_index_tensor = src_index.view(-1,1).to(src_point.device)