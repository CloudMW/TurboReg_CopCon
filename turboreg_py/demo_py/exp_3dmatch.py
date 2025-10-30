import time
from typing import Literal
import tyro
import torch
from dataclasses import dataclass

from turboreg_py.demo_py.dataset_3dmatch import TDMatchFCGFAndFPFHDataset
from turboreg_py.demo_py.utils_pcr import compute_transformation_error, numpy_to_torch32, numpy_to_torchint32
from turboreg_py import TurboRegGPU

@dataclass
class Args:
    # Dataset path
    dir_dataset: str
    desc: Literal["fpfh", "fcgf"] = "fcgf"
    dataname: Literal["3DMatch", "3DLoMatch"] = "3DLoMatch"

    # TurboRegGPU Initialization Parameters
    max_N: int = 7000
    tau_length_consis: float = 0.012
    num_pivot: int = 2000
    radiu_nms: float = 0.15
    tau_inlier: float = 0.1
    metric_str: Literal["IN", "MSE", "MAE"] = "IN"


from sklearn.neighbors import KDTree  # 需安装 scikit-learn
def find_keypoint_indices_kdtree(original_points, keypoints, eps=1e-6):
    """
    基于KDTree的快速搜索，几乎不占用额外内存

    注意：需将张量转移到CPU（scikit-learn的KDTree不支持GPU）
    """
    # 转为CPU numpy数组（KDTree不支持GPU张量）
    original_np = original_points.cpu().numpy()
    keypoints_np = keypoints.cpu().numpy()

    # 构建原始点云的KDTree
    kdtree = KDTree(original_np)

    # 对每个关键点搜索最近邻（返回距离和索引）
    distances, indices = kdtree.query(keypoints_np, k=1)  # k=1表示只找最近的1个点

    # 转换为torch张量，过滤距离超过eps的点
    indices = torch.from_numpy(indices.squeeze()).to(original_points.device)  # [M]
    distances = torch.from_numpy(distances.squeeze()).to(original_points.device)  # [M]

    indices[distances >= eps] = -1  # 无效匹配设为-1
    return indices

def main(device):
    args = tyro.cli(Args)

    if args.dataname.lower() == "3dmatch":
        processed_dataname = "3DMatch"
    elif args.dataname.lower() == "3dlomatch":
        processed_dataname = "3DLoMatch"
    else:
        raise ValueError(f"Invalid dataname: {args.dataname}. Expected '3DMatch' or '3DLoMatch'.")

    # TurboReg
    reger = TurboRegGPU(
        args.max_N,
        args.tau_length_consis,
        args.num_pivot,
        args.radiu_nms,
        args.tau_inlier,
        args.metric_str
    )

    ds = TDMatchFCGFAndFPFHDataset(base_dir=args.dir_dataset, dataset_type=processed_dataname, descriptor_type=args.desc)

    num_succ = 0
    for i in range(1400,len(ds)):
        data = ds[i]

        # "corr_kpts_src": corr_kpts_src,
        # "corr_kpts_dst": corr_kpts_dst,
        # "trans_gt": trans_gt,
        # "pts_src": src_cloud,
        # "pts_dst": dst_cloud,
        # "kpts_src": kpts_src,
        # "kpts_dst": kpts_dst,
        # "corr_ind": corr_ind
        # kpts_src, kpts_dst, trans_gt ,pts_src,pts_dst= data['kpts_src'], data['kpts_dst'], data['trans_gt'],data['pts_src'],data['pts_dst']

        corr_kpts_src ,corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,corr_ind = data['corr_kpts_src'], data['corr_kpts_dst'], data['trans_gt'], data['pts_src'], data['pts_dst'], data['kpts_src'], data['kpts_dst'], data['corr_ind']
        # Move keypoints to CUDA device
        corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst = numpy_to_torch32(
            device,  corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst
        )
        [corr_ind] = numpy_to_torchint32(device,corr_ind)
        # src_ind = find_keypoint_indices_kdtree(pts_src, kpts_src, eps=1e-6)
        # dst_ind = find_keypoint_indices_kdtree(pts_dst, kpts_dst, eps=1e-6)

        # corr_ind = torch.stack([src_ind, dst_ind], dim=1)  # [M, 2]
        # Run TurboReg
        t1 = time.time()
        trans_pred_torch = reger.run_reg(corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,corr_ind )
        T_reg = (time.time() - t1) * 1000
        trans_pred = trans_pred_torch.cpu().numpy()
        trans_gt = trans_gt.cpu().numpy()
        rre, rte = compute_transformation_error(trans_gt, trans_pred)
        is_succ = (rre < 15) & (rte < 0.3)
        if not is_succ:
            print("Registration failed for item {}/{}: RRE={:.3f}, RTE={:.3f}".format().format(i+1, len(ds), rre, rte))
        num_succ += is_succ
        
        print(f"Processed item {i+1}/{len(ds)}: Registration time: {T_reg:.3f} ms, RR= {(num_succ / (i+1)) * 100:.3f}%")

if __name__ == "__main__":
    import os
    def is_debugging():
        # 检测常见调试器环境变量
        debug_env_vars = [
            "PYDEVD_LOAD_VALUES_ASYNC",  # PyCharm/VS Code 调试器
            "DEBUGPY_DEBUG_MODE",  # VS Code 调试器
            "PYCHARM_DEBUG",  # PyCharm 调试器
        ]
        return any(var in os.environ for var in debug_env_vars)


    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for computations.")
        device = torch.device("cuda:0")
    elif not is_debugging() and torch.xpu.is_available():
        print("XPU is available. Using XPU for computations.")
        device = torch.device("xpu:0")
    else:
        print("CUDA is not available. Using CPU for computations.")
        device = torch.device("cpu")
    main(device)

"""
python -m demo_py.exp_3dmatch --desc fpfh --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 7000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.15 --tau_inlier 0.1 --metric_str "IN"
python -m demo_py.exp_3dmatch --desc fcgf --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 6000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.10 --tau_inlier 0.1 --metric_str "MAE"
"""