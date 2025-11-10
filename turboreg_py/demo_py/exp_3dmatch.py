import time
from typing import Literal
import tyro
import torch
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

from turboreg_py.demo_py.dataset_3dmatch import TDMatchFCGFAndFPFHDataset
from turboreg_py.demo_py.utils_pcr import compute_transformation_error, numpy_to_torch32, numpy_to_torchint32
from turboreg_py import TurboRegGPU
from turboreg_py import TurboRegPlus

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

    # Result saving and rerun options
    out_root: str = "result"  # root folder to save results: <out_root>/<dataset>/<desc>/
    rerun_errors: bool = False  # if True, only rerun indices listed in error_result.txt


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


def _ensure_result_paths(out_root: str, dataset: str, desc: str) -> Path:
    """Create and return result directory path: out_root/dataset/desc"""
    path = Path(out_root) / dataset / desc
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_result_file(file_path: Path, index: int, rre: float, rte: float, transform: 'np.ndarray') -> None:
    """Append a single result (5 lines) to file: first line index rre rte, next 4 lines transform matrix rows."""
    with open(file_path, "a") as f:
        f.write(f"{index} {rre:.6f} {rte:.6f}\n")
        # transform is expected shape (4,4)
        for row in transform:
            f.write(" ".join([f"{val:.8f}" for val in row]) + "\n")


def _read_error_indices(file_path: Path) -> list:
    """Read error_result.txt and return list of indices (first number of each 5-line block)."""
    if not file_path.exists():
        return []
    indices = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    # parse blocks of 5 lines: more explicit iteration to satisfy static checks
    for j in range(0, len(lines), 5):
        header_line = lines[j].strip()
        if not header_line:
            continue
        header = header_line.split()
        if not header:
            continue
        try:
            idx = int(header[0])
            indices.append(idx)
        except Exception:
            # ignore malformed header
            continue
    return indices


def main(device):
    args = tyro.cli(Args)

    if args.dataname.lower() == "3dmatch":
        processed_dataname = "3DMatch"
    elif args.dataname.lower() == "3dlomatch":
        processed_dataname = "3DLoMatch"
    else:
        raise ValueError(f"Invalid dataname: {args.dataname}. Expected '3DMatch' or '3DLoMatch'.")

    # Prepare result paths
    result_dir = _ensure_result_paths(args.out_root, processed_dataname, args.desc)
    all_result_file = result_dir / "all_result.txt"
    error_result_file = result_dir / "error_result.txt"
    log_file = result_dir / "log.txt"

    # If this is a normal run (not rerun_errors), clear any previous results so we start fresh
    if not args.rerun_errors:
        for p in (all_result_file, error_result_file, log_file):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # ignore potential race/permission errors here; files will be re-created on write
                pass

    # If rerun_errors mode is enabled, read indices to run
    indices_to_run = None
    if args.rerun_errors:
        indices_to_run = _read_error_indices(error_result_file)
        if len(indices_to_run) == 0:
            print(f"No error indices found in {error_result_file}. Running full dataset instead.")
            indices_to_run = None
        else:
            print(f"Rerun mode: will process {len(indices_to_run)} items from error file.")

    # TurboReg
    reger = TurboRegGPU(
        args.max_N,
        args.tau_length_consis,
        args.num_pivot,
        args.radiu_nms,
        args.tau_inlier,
        args.metric_str
    )
    reger_plus = TurboRegPlus(
        args.max_N,
        args.tau_length_consis,
        args.num_pivot,
        args.radiu_nms,
        args.tau_inlier,
        args.metric_str
    )

    ds = TDMatchFCGFAndFPFHDataset(base_dir=args.dir_dataset, dataset_type=processed_dataname, descriptor_type=args.desc)

    # run error

    num_succ = 0
    total = 0
    rr_list = []  # successful RREs
    re_list = []  # successful RTEs

    # If indices_to_run is provided, iterate only those items (assume they are 0-based indices)
    if indices_to_run is not None:
        iterable = indices_to_run
    else:
        iterable = range(0, len(ds))

    t_all = time.time()
    for loop_i, idx in enumerate(iterable):
        i = int(idx)
        data = ds[i]

        corr_kpts_src ,corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,corr_ind,feature_kpts_src,feature_kpts_dst =\
            (data['corr_kpts_src'], data['corr_kpts_dst'], data['trans_gt'],
             data['pts_src'], data['pts_dst'], data['kpts_src'], data['kpts_dst'], data['corr_ind'],data['feature_kpts_src'],data['feature_kpts_dst'])
        # Move keypoints to CUDA device
        if feature_kpts_src is None:
            corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst= numpy_to_torch32(
                device,  corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst
            )
        else :
            corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,feature_kpts_src,feature_kpts_dst= numpy_to_torch32(
                device,  corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,feature_kpts_src,feature_kpts_dst
            )
        [corr_ind] = numpy_to_torchint32(device,corr_ind)

       # Run TurboReg
        t1 = time.time()
        trans_pred_torch = reger.run_reg(corr_kpts_src, corr_kpts_dst,trans_gt,src_cloud,dst_cloud,kpts_src,kpts_dst,corr_ind,feature_kpts_src,feature_kpts_dst )
        T_reg = (time.time() - t1) * 1000
        trans_pred = trans_pred_torch.cpu().numpy()
        trans_gt_numpy = trans_gt.cpu().numpy()
        rre, rte = compute_transformation_error(trans_gt_numpy, trans_pred)
        is_succ = (rre < 15) & (rte < 0.3)
        print("TurboReg Result: RRE={:.6f}, RTE={:.6f}, Success={}".format(rre, rte, is_succ))


        # Run TurboRegPlus
        t_regor_plus = time.time()
        trans_pred_torch_plus = reger_plus.run_reg(corr_kpts_src, corr_kpts_dst, trans_gt, src_cloud, dst_cloud, kpts_src,
                                         kpts_dst, corr_ind, feature_kpts_src, feature_kpts_dst)
        T_reg_plus = (time.time() - t_regor_plus) * 1000
        trans_pred_plus = trans_pred_torch_plus.cpu().numpy()
        trans_gt_numpy = trans_gt.cpu().numpy()
        rre, rte = compute_transformation_error(trans_gt_numpy, trans_pred_plus)
        is_succ = (rre < 15) & (rte < 0.3)
        print("TurboRegPlus Result: RRE={:.6f}, RTE={:.6f}, Success={}".format(rre, rte, is_succ))

        # save result: append to all_result
        if not args.rerun_errors:
            _append_result_file(all_result_file, i, float(rre), float(rte), trans_pred)
            if not is_succ:
                _append_result_file(error_result_file, i, float(rre), float(rte), trans_pred)

        total += 1
        num_succ += int(is_succ)
        if is_succ:
            rr_list.append(float(rre))
            re_list.append(float(rte))
        else:
            print(f"Registration failed for item {i}/{len(ds)-1}: RRE={rre}, RTE={rte}")
        print(f"Processed item {loop_i+1}/{len(iterable)} (dataset idx {i}): Registration time: {T_reg:.3f} ms, RR= {(num_succ / total) * 100:.3f}%")

    t_all = time.time() - t_all

    # After processing, write log
    recall = (num_succ / total) * 100 if total > 0 else 0.0
    avg_rr = sum(rr_list) / len(rr_list) if len(rr_list) > 0 else 0.0
    avg_re = sum(re_list) / len(re_list) if len(re_list) > 0 else 0.0



    if not args.rerun_errors:
        with open(log_file, "w") as f:
            f.write(f"Total: {total}\n")
            f.write(f"Successful: {num_succ}\n")
            f.write(f"Recall(%%): {recall:.3f}\n")
            f.write(f"Avg_RRE_of_correct: {avg_rr:.6f}\n")
            f.write(f"Avg_RTE_of_correct: {avg_re:.6f}\n")
            f.write(f"Total_time_seconds: {t_all:.3f}\n")

        print(f"Done. Results saved in {result_dir}. Recall={recall:.3f}%, Avg_RRE={avg_rr:.6f}, Avg_RTE={avg_re:.6f}")
        print(f"Total time: {t_all:.3f} seconds for {total} items.")
    else:
        # In rerun-errors mode we skip writing any result files
        print(f"Done. Rerun-errors mode: processed {total} items (no result files written). Recall={recall:.3f}%, Avg_RRE={avg_rr:.6f}, Avg_RTE={avg_re:.6f}")


if __name__ == "__main__":
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
    else:
        # If not debugging, try XPU (if present); otherwise fall back to CPU
        if not is_debugging():
            xpu = getattr(torch, 'xpu', None)
            if xpu is not None and xpu.is_available():
                print("XPU is available. Using XPU for computations.")
                device = torch.device("xpu:0")
            else:
                print("CUDA/XPU not available. Using CPU for computations.")
                device = torch.device("cpu")
        else:
            print("CUDA is not available. Using CPU for computations.")
            device = torch.device("cpu")
    main(device)

"""
python -m demo_py.exp_3dmatch --desc fpfh --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 7000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.15 --tau_inlier 0.1 --metric_str "IN"
python -m demo_py.exp_3dmatch --desc fcgf --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 6000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.10 --tau_inlier 0.1 --metric_str "MAE"
"""