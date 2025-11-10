import torch
from torch.cuda import device


def log_sinkhorn_normalization(scores, log_mu, log_nu):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(10):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)

def neighbors(points: torch.Tensor, neighbors_num: int) -> torch.Tensor:
    """
    For each point in `points`, return indices of `neighbors_num` nearest neighbors (excluding the point itself).

    Args:
        points: [N, 3] tensor of point coordinates (float)
        neighbors_num: number of neighbors to return per point

    Returns:
        indices: LongTensor of shape [N, neighbors_num] with indices into the first dimension of `points`.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)

    device = points.device
    N = points.shape[0]

    # clamp neighbors_num to at most N-1 (exclude self)
    k = min(max(0, neighbors_num), max(0, N - 1))

    if k == 0:
        # return an empty tensor with correct shape
        return torch.empty((N, 0), dtype=torch.long, device=device)

    # Compute pairwise distances (uses optimized CUDA kernels when available)
    # Use squared distances for efficiency
    # cdist returns [N, N]
    dists = torch.cdist(points, points, p=2)  # float tensor [N, N]

    # Mask self-distance to avoid selecting the point itself
    idx = torch.arange(N, device=device)
    dists[idx, idx] = float('inf')

    # Get k smallest distances per row
    _, knn_idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)

    return knn_idx

def optimal_transport( src_points,tgt_point,src_feature,tgt_feature):
    r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

    Args:
        scores: torch.Tensor (B, M, N)
        row_masks: torch.Tensor (B, M)
        col_masks: torch.Tensor (B, N)

    Returns:
        matching_scores: torch.Tensor (B, M+1, N+1)
    """
    # device = src_points.device
    # src_p = src_points.shape[0]
    #
    # corr = get_corr(src_feature,tgt_feature)
    #
    # src_neighbor_idx = neighbors(src_points, neighbors_num=32)  # (N_src, K)
    # tgt_neighbor_idx = neighbors(tgt_point, neighbors_num=32)
    #
    # src_neighbor_feat = src_feature[src_neighbor_idx]  # (N_src, K, C)
    # tgt_neighbor_feat = tgt_feature[tgt_neighbor_idx] [corr[:,1]]# (N_tgt, K, C)
    #
    #
    # # 计算节点内K近邻点之间的特征相似度矩阵
    # matching_scores = torch.einsum('bnd,bmd->bnm', src_neighbor_feat, tgt_neighbor_feat)  # (P, K, K)
    # # 特征维度归一化，提高数值稳定性
    # # scores = matching_scores / src_p ** 0.5 # feats_f.shape[1] = B
    # scores = matching_scores
    #
    #
    # batch_size, num_row, num_col = scores.shape
    # inf = 1e12
    #
    # # 默认所有点都有效（如需部分无效，可传入 masks）
    # row_masks = torch.ones((batch_size, num_row), dtype=torch.bool, device=device)
    # col_masks = torch.ones((batch_size, num_col), dtype=torch.bool, device=device)
    #
    # # 生成 2D mask：行或列任意无效则该位置被 mask
    # score_mask = torch.logical_or(~row_masks.unsqueeze(2), ~col_masks.unsqueeze(1))
    # scores = scores.masked_fill(score_mask, -inf)
    #
    # # 计算有效点数量与归一化因子
    # num_valid_row = row_masks.float().sum(1)  # (B,)
    # num_valid_col = col_masks.float().sum(1)  # (B,)
    # norm = -torch.log(num_valid_row + num_valid_col)  # (B,)
    #
    # # 构造边际分布 log_mu, log_nu（不包含 dustbin）
    # log_mu = norm.unsqueeze(1).expand(batch_size, num_row).clone()
    # log_nu = norm.unsqueeze(1).expand(batch_size, num_col).clone()
    #
    # # 将无效行/列对应的边际设为 -inf
    # log_mu[~row_masks] = -inf
    # log_nu[~col_masks] = -inf
    #
    # # 执行 Sinkhorn（不含 dustbin）
    # outputs = log_sinkhorn_normalization(scores, log_mu, log_nu)
    # outputs = outputs - norm.unsqueeze(1).unsqueeze(2)
    n_iters = 20

    # S = src_feature @ tgt_feature.T
    # eps = 0.05
    # M, N = S.shape
    # log_K = S / eps  # pass directly (optionally subtract max for stability per-row/whole)
    # # target marginals: uniform mass over rows/cols (sum to 1)
    # r = torch.ones(M, device=S.device) / M
    # c = torch.ones(N, device=S.device) / N
    #
    # P, iters = sinkhorn_log_domain(log_K, r, c, max_iters=200, tol=1e-2, verbose=True)
    # src_idx,tgt_idx,_,_=top_overlap_points(P, ratio=1/3, mode='max')
    src_neighbor_idx = neighbors(src_points, neighbors_num=50)  # (N_src, K)
    tgt_neighbor_idx = neighbors(tgt_point, neighbors_num=50)

    src_neighbor_feat = src_feature[src_neighbor_idx]  # (N_src, K, C)
    tgt_neighbor_feat = tgt_feature[tgt_neighbor_idx]
    src_idx,tgt_idx,_,_,_ = find_overlap_points_with_ot(src_points,tgt_point,src_feature,tgt_feature,src_neighbor_feat,tgt_neighbor_feat,num_iters = 20)
    return src_idx,tgt_idx

import torch.nn.functional as F
def find_overlap_points_with_ot(points_A, points_B, FA, FB, FA_neighbor, FB_neighbor,
                                k_overlap=500, num_iters=20, temperature=0.05):
    """
    基于邻域特征和最优传输的点云重叠检测

    参数:
        points_A: [M, 3] 点云A的坐标
        points_B: [N, 3] 点云B的坐标
        FA: [M, 32] 点云A的特征
        FB: [N, 32] 点云B的特征
        FA_neighbor: [M, D, 32] 点云A每个点的D个邻域点特征
        FB_neighbor: [N, D, 32] 点云B每个点的D个邻域点特征
        k_overlap: 期望保留的重叠点数量
        num_iters: Sinkhorn迭代次数
        temperature: 温度参数，控制匹配的软硬程度

    返回:
        overlap_A: [K1, 3] 点云A中重叠概率高的点
        overlap_B: [K2, 3] 点云B中重叠概率高的点
        scores_A: [K1] A中每个重叠点的得分
        scores_B: [K2] B中每个重叠点的得分
    """
    M, D, feat_dim = FA_neighbor.shape
    N = FB_neighbor.shape[0]
    device = FA.device

    # 步骤1: 聚合邻域特征，构建上下文描述符
    # 使用注意力机制聚合邻域
    FA_center = FA.unsqueeze(1)  # [M, 1, 32]
    FB_center = FB.unsqueeze(1)  # [N, 1, 32]

    # 计算中心点与邻域的注意力权重

    attn_A = torch.sum(FA_center * FA_neighbor, dim=-1)  # [M, D]
    attn_A = F.softmax(attn_A / (feat_dim ** 0.5), dim=-1)  # [M, D]

    attn_B = torch.sum(FB_center * FB_neighbor, dim=-1)  # [N, D]
    attn_B = F.softmax(attn_B / (feat_dim ** 0.5), dim=-1)  # [N, D]

    # 加权聚合邻域特征
    FA_context = torch.sum(attn_A.unsqueeze(-1) * FA_neighbor, dim=1)  # [M, 32]
    FB_context = torch.sum(attn_B.unsqueeze(-1) * FB_neighbor, dim=1)  # [N, 32]

    # 拼接中心特征和上下文特征
    FA_enhanced = torch.cat([FA, FA_context], dim=-1)  # [M, 64]
    FB_enhanced = torch.cat([FB, FB_context], dim=-1)  # [N, 64]

    # L2归一化
    FA_enhanced = F.normalize(FA_enhanced, p=2, dim=-1)
    FB_enhanced = F.normalize(FB_enhanced, p=2, dim=-1)

    # 步骤2: 计算成本矩阵 (使用余弦距离)
    # 成本 = 1 - 余弦相似度
    similarity = torch.matmul(FA_enhanced, FB_enhanced.t())  # [M, N]
    cost_matrix = 1.0 - similarity  # [M, N]

    # 步骤3: Sinkhorn最优传输
    # 初始化
    log_P = -cost_matrix / temperature  # [M, N]

    # Sinkhorn迭代
    for _ in range(num_iters):
        # 行归一化
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        # 列归一化
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)

    P = torch.exp(log_P)  # [M, N] 传输矩阵

    # 步骤4: 计算每个点的重叠得分
    # 对于A中的点: 其在B中的最大传输概率
    overlap_score_A = torch.max(P, dim=1)[0]  # [M]

    # 对于B中的点: 其在A中的最大传输概率
    overlap_score_B = torch.max(P, dim=0)[0]  # [N]

    # 步骤5: 根据得分筛选重叠点
    # 对A进行筛选
    k_A = min(points_A.shape[0]//3, M)
    topk_A = torch.topk(overlap_score_A, k=k_A)
    indices_A = topk_A.indices  # [K1]
    scores_A = topk_A.values  # [K1]

    # 对B进行筛选
    k_B = min(points_B.shape[0]//3, N)
    topk_B = torch.topk(overlap_score_B, k=k_B)
    indices_B = topk_B.indices  # [K2]
    scores_B = topk_B.values  # [K2]

    # 提取重叠点云
    overlap_A = points_A[indices_A]  # [K1, 3]
    overlap_B = points_B[indices_B]  # [K2, 3]

    return indices_A, indices_B, scores_A, scores_B, P
def top_overlap_points(P, ratio=1/3, mode='max'):
    """
    从 Sinkhorn 概率矩阵中选出重叠概率最高的前若干点。
    Args:
        P: [M, N] Sinkhorn 概率矩阵
        ratio: 选取比例 (0~1)
        mode: 'max' 或 'sum'，控制使用哪种方式计算重叠概率
    Returns:
        topA_idx: Tensor[int]，点云A的高重叠点索引
        topB_idx: Tensor[int]，点云B的高重叠点索引
        pA, pB: 各点的重叠概率
    """
    # 计算各点的重叠概率
    if mode == 'max':
        pA, _ = P.max(dim=1)   # [M]
        pB, _ = P.max(dim=0)   # [N]
    elif mode == 'sum':
        pA = P.sum(dim=1)
        pB = P.sum(dim=0)
    else:
        raise ValueError("mode 必须是 'max' 或 'sum'")

    # 选取前 ratio 百分比的点
    M, N = P.shape
    kA = int(M * ratio)
    kB = int(N * ratio)

    topA_val, topA_idx = torch.topk(pA, k=kA)
    topB_val, topB_idx = torch.topk(pB, k=kB)

    return topA_idx, topB_idx, pA, pB
def sinkhorn_log_domain(log_K, r, c, max_iters=200, tol=1e-3, verbose=False):
    """
    Log-domain Sinkhorn to compute P ~ diag(u) K diag(v), where K = exp(S/eps).
    Inputs:
      - log_K: [M, N] tensor of log(K) (so you can pass (S/eps) with numerical stability)
      - r: [M] target row sums (usually 1/M or actual masses)
      - c: [N] target col sums (usually 1/N)
      - max_iters: maximum iterations
      - tol: L1 tolerance on marginal error for stopping
    Returns:
      - P: [M,N] probability matrix (dense) or returns logP if you prefer
      - iters: actual number of iterations
    Notes:
      - Using log-space avoids under/overflow when eps small.
      - r and c should sum to same total (e.g., both sum to 1).
    """
    M, N = log_K.shape
    device = log_K.device
    # initialize dual potentials (log u, log v) as zeros
    log_u = torch.zeros(M, device=device)
    log_v = torch.zeros(N, device=device)

    # helper for row/col sums from log-domain
    def log_sum_exp_axis(mat, dim):
        # mat: tensor, dim: axis to logsumexp over
        return torch.logsumexp(mat, dim=dim)

    for i in range(1, max_iters + 1):
        # update log_u: ensure rows sum to r -> log_u = log r - logsumexp(log_K + log_v)
        # note broadcasting: log_K + log_v (N) adds log_v to each row
        log_u = torch.log(r) - log_sum_exp_axis(log_K + log_v.unsqueeze(0), dim=1)

        # update log_v: ensure columns sum to c -> log_v = log c - logsumexp(log_K^T + log_u)
        log_v = torch.log(c) - log_sum_exp_axis(log_K.t() + log_u.unsqueeze(0), dim=1)

        if (i % 10 == 0) or i == max_iters:
            # compute current P marginals in log-domain to check error
            # logP = log_u[:,None] + log_K + log_v[None,:]
            logP = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
            # get marginals
            row_logsum = torch.logsumexp(logP, dim=1)  # log row sums
            col_logsum = torch.logsumexp(logP, dim=0)  # log col sums
            row_sum = torch.exp(row_logsum)
            col_sum = torch.exp(col_logsum)
            err = torch.sum(torch.abs(row_sum - r)) + torch.sum(torch.abs(col_sum - c))
            if verbose:
                print(f"iter {i:03d} marginal L1 error {err.item():.3e}")
            if err.item() < tol:
                break

    # final P
    logP = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(logP)
    return P, i

def get_corr(src_desc,tgt_desc):
    distance = torch.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    source_idx = torch.argmin(distance, axis=1)
    use_mutual = False
    if use_mutual:
        target_idx = torch.argmin(distance, axis=0)
        mutual_nearest = (target_idx[source_idx] == torch.arange(source_idx.shape[0]).to(source_idx.device))
        corr = torch.concatenate([torch.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
                              axis=-1)
    else:
        corr = torch.concatenate([torch.arange(source_idx.shape[0]).to(source_idx.device)[:, None], source_idx[:, None]], axis=-1)

    return corr
