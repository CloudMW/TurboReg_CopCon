import numpy as np
import maxflow
def graph_cut_select(C2, desc_score=None, lambda_pair=0.3):
    """
    C2: NxN compatibility matrix
    desc_score: Nx1 single-node confidence (e.g., feature similarity)
    """
    N = C2.shape[0]
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(N)

    # 单点项 D_i(x_i)
    if desc_score is None:
        desc_score = np.ones(N)
    desc_score = (desc_score - desc_score.min()) / (desc_score.max() - desc_score.min() + 1e-6)

    for i in range(N):
        # 连接源点（选中代价）和汇点（不选代价）
        g.add_tedge(nodes[i],
                    1 - desc_score[i],  # to source
                    desc_score[i])      # to sink

    # 成对项 λ * (1 - C_ij)
    for i in range(N):
        for j in range(i + 1, N):
            w = lambda_pair * (1 - C2[i, j])
            if w > 0:
                g.add_edge(nodes[i], nodes[j], w, w)

    flow = g.maxflow()
    selected = np.array([g.get_segment(nodes[i]) == 0 for i in range(N)])  # 0=source side

    return selected
import torch
def graph_cut_seed(corr_kpts_src, corr_kpts_dst, hard=False,tau_length_consis=0.012):
    M,_ = corr_kpts_dst.shape

    radiu_nms = 0.1
    src_dist = torch.norm(
        corr_kpts_src.unsqueeze(1) - corr_kpts_src.unsqueeze(0),
        p=2, dim=-1
    )  # [N, N]
    target_dist = torch.norm(
        corr_kpts_dst.unsqueeze(1) - corr_kpts_dst.unsqueeze(0),
        p=2, dim=-1
    )  # [N, N]

    cross_dist = torch.abs(src_dist - target_dist)  # [N, N]

    # Compute compatibility
    SC_dist_thre = 0.1
    if not hard:
        C2 = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
    else:
        C2 = (cross_dist < 0.1).float()

    # Apply mask based on distance threshold (NMS)
    mask = (src_dist + target_dist) <= radiu_nms
    C2 = C2.masked_fill(mask, 0)
    SC2 = torch.matmul(C2, C2) * C2
    selected = graph_cut_select(SC2.cpu().numpy(), desc_score=None)
    seed_mask = torch.tensor(selected, device=C2.device)
    seed_src = corr_kpts_src[seed_mask]
    seed_dst = corr_kpts_dst[seed_mask]
    return seed_src, seed_dst, seed_mask