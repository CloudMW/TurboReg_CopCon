import torch
def find_maximal_cliques_dynamic_robust(SC2, M, A, device='cuda', min_clique_size=3):
    """
    改进版：处理无法扩展的团

    参数:
        SC2: [N, N] 邻接矩阵
        M: 需要找到的极大团数量
        A: 目标团大小
        device: 计算设备
        min_clique_size: 最小可接受的团大小

    返回:
        cliques_tensor: [M, A] 团列表，无效位置用-1填充
        valid_sizes: [M] 每个团的实际大小
    """
    N = SC2.size(0)

    # Step 1: 找到前M个权重最大的边
    SC2_up = torch.triu(SC2, diagonal=1)
    flat_SC2_up = SC2_up.flatten()
    scores_topk, idx_topk = torch.topk(flat_SC2_up, M)

    pivots = torch.stack([
        (idx_topk // N).long(),
        (idx_topk % N).long()
    ], dim=1)

    # 初始化团张量，用-1填充（表示无效）
    cliques = torch.full((M, A), -1, dtype=torch.long, device=device)
    cliques[:, :2] = pivots

    # 记录每个团的实际大小
    valid_sizes = torch.full((M,), 2, dtype=torch.long, device=device)

    # 标记哪些团还可以继续扩展
    active_mask = torch.ones(M, dtype=torch.bool, device=device)

    SC2_search = SC2.clone()

    # Step 2: 逐步扩展团
    for current_size in range(2, A):
        if not active_mask.any():
            break  # 所有团都无法继续扩展

        # 只处理还活跃的团
        active_indices = torch.where(active_mask)[0]
        num_active = active_indices.size(0)

        if num_active == 0:
            break

        # 获取活跃团的当前顶点
        current_clique = cliques[active_indices][:, :current_size]

        # 找到候选顶点
        candidate_indicator = torch.ones((num_active, N), dtype=torch.bool, device=device)

        for i in range(current_size):
            vertex_indices = current_clique[:, i]
            adjacency = SC2_search[vertex_indices] > 0
            candidate_indicator = candidate_indicator & adjacency

        # 排除已在团中的顶点
        for i in range(current_size):
            vertex_indices = current_clique[:, i]
            candidate_indicator[torch.arange(num_active, device=device), vertex_indices] = False

        # 计算得分
        scores = torch.zeros((num_active, N), dtype=SC2.dtype, device=device)
        for i in range(current_size):
            vertex_indices = current_clique[:, i]
            scores += SC2_search[vertex_indices]

        scores = scores * candidate_indicator.float()

        # 检查哪些团还有合法候选
        has_candidates = candidate_indicator.any(dim=1)

        # 更新活跃掩码
        for idx, (active_idx, has_cand) in enumerate(zip(active_indices, has_candidates)):
            if not has_cand:
                active_mask[active_idx] = False
            else:
                # 选择最佳候选
                best_candidate = torch.argmax(scores[idx])
                cliques[active_idx, current_size] = best_candidate
                valid_sizes[active_idx] = current_size + 1

    return cliques, valid_sizes


def filter_valid_cliques(cliques, valid_sizes, min_size=3):
    """
    过滤并返回有效的团

    返回:
        valid_cliques: 有效团的列表（不定长）
        valid_sizes: 对应的大小
    """
    mask = valid_sizes >= min_size
    valid_indices = torch.where(mask)[0]

    result_cliques = []
    result_sizes = []

    for idx in valid_indices:
        size = valid_sizes[idx].item()
        clique = cliques[idx, :size]
        result_cliques.append(clique)
        result_sizes.append(size)

    return result_cliques, result_sizes
