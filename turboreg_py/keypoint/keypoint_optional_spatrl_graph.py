import torch
from turboreg_py.keypoint.seed_point import get_seed
from turboreg_py.keypoint.keypoints_optimal_transport import optimal_transport
from turboreg_py.keypoint.keypoint import get_keypoint_from_scores
from turboreg_py.keypoint.graph_cut import graph_cut_seed

def min_max_normalize(x, dim=-1):
    """
    Min-Max归一化：缩放到[0,1]
    Args:
        x: 输入向量 (bs, N) 或 (N,)
        dim: 归一化维度（默认-1，即最后一维，适配批次）
    Returns:
        x_norm: 归一化后的向量，形状与输入一致
    """
    x_min = torch.min(x, dim=dim, keepdim=True)[0]  # 计算最小值（保留维度，方便广播）
    x_max = torch.max(x, dim=dim, keepdim=True)[0]  # 计算最大值
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)   # 归一化
    return x_norm
def keypoint_spectral_graph(src_point,tgt_point,feature_kpts_src,feature_kpts_dst,is_FCGF=False):
    corr_ind_src_2_tgt = get_corr(feature_kpts_src, feature_kpts_dst)
    score_src_sg = get_seed(src_point, tgt_point[corr_ind_src_2_tgt[:, 1]])
    corr_ind_tgt_2_src = get_corr(feature_kpts_dst, feature_kpts_src)
    score_tgt_sg= get_seed(tgt_point, src_point[corr_ind_tgt_2_src[:, 1]])
    score_src_opt , score_tgt_opt = optimal_transport(src_point,tgt_point,feature_kpts_src,feature_kpts_dst)
    score_src_sg = min_max_normalize(score_src_sg.squeeze(0))
    score_tgt_sg = min_max_normalize(score_tgt_sg.squeeze(0))
    score_src_d3feat = get_keypoint_from_scores(src_point,feature_kpts_src).squeeze(-1)
    score_tgt_d3feat = get_keypoint_from_scores(tgt_point,feature_kpts_dst).squeeze(-1)
    # score_src_graph_cut = graph_cut_seed(src_point, tgt_point[corr_ind_src_2_tgt[:, 1]])

    score_src_d3feat = min_max_normalize(score_src_d3feat)
    score_tgt_d3feat = min_max_normalize(score_tgt_d3feat)
    score_src_opt = min_max_normalize(score_src_opt)
    score_tgt_opt = min_max_normalize(score_tgt_opt)
    score_src = score_src_sg
    score_tgt = score_tgt_sg
    _,src_index = score_src.topk(k=src_point.shape[0]//3,dim=0)
    _,tgt_index = score_tgt.topk(k=tgt_point.shape[0]//3,dim=0)
    return src_index,tgt_index
def get_corr( src_desc, tgt_desc,is_FCGF= False):
    distance = torch.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    source_idx = torch.argmin(distance, axis=1)
    use_mutual = False

    if use_mutual:
        target_idx = torch.argmin(distance, axis=0)
        mutual_nearest = (target_idx[source_idx] == torch.arange(source_idx.shape[0]).to(source_idx.device))
        corr = torch.concatenate(
            [torch.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
            axis=-1)
    else:
        corr = torch.concatenate([torch.arange(source_idx.shape[0]).to(source_idx.device)[:, None], source_idx[:, None]], axis=-1)

    return corr