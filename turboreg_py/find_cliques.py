import torch


def find_maximal_cliques_dynamic_robust(SC2, M, A, device='cuda',  X=None):
    """
    优化版：在搜索过程中抛弃不符合条件的锚点，并返回top-X个极大团

    参数:
        SC2: [N, N] 邻接矩阵
        M: 锚点数量（初始pivot数量）
        A: 目标极大团大小
        device: 计算设备
        min_clique_size: 最小可接受的团大小
        X: 返回的极大团数量（如果为None，则返回所有符合条件的团）

    返回:
        cliques_tensor: [X, A] 或 [num_valid, A] 团列表
        valid_sizes: [X] 或 [num_valid] 每个团的实际大小
    """
    N = SC2.size(0)

    # 如果X未指定，默认返回所有符合条件的团
    if X is None:
        X = M

    # Step 1: 找到前M个权重最大的边作为锚点
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

    # 标记哪些团还可以继续扩展（初始所有锚点都是活跃的）
    active_mask = torch.ones(M, dtype=torch.bool, device=device)

    SC2_search = SC2.clone()

    # Step 2: 逐步扩展团，从大小2扩展到大小A
    for current_size in range(2, A):
        if not active_mask.any():
            break  # 所有团都已被抛弃

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
        has_candidates = candidate_indicator.any(dim=1)  # [num_active]

        # 批量更新活跃掩码：如果找不到候选顶点，立即抛弃该锚点
        # 对于没有候选的锚点，将其从活跃掩码中移除
        active_mask[active_indices[~has_candidates]] = False

        # 批量选择最佳候选（只对有候选的团进行处理）
        if has_candidates.any():
            # 找到有候选的团的索引
            valid_active_mask = has_candidates  # [num_active]
            valid_active_indices = active_indices[valid_active_mask]  # 有候选的活跃团的全局索引
            valid_scores = scores[valid_active_mask]  # [num_valid_active, N]

            # 批量找到最佳候选
            best_candidates = torch.argmax(valid_scores, dim=1)  # [num_valid_active]

            # 批量更新团和大小
            cliques[valid_active_indices, current_size] = best_candidates
            valid_sizes[valid_active_indices] = current_size + 1

    # Step 3: 过滤出达到目标大小A的团
    valid_mask = valid_sizes >= A
    valid_indices = torch.where(valid_mask)[0]
    num_valid = valid_indices.size(0)

    if num_valid == 0:
        # 没有找到符合条件的团，返回空结果
        return torch.empty((0, A), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    # 提取有效的团
    valid_cliques = cliques[valid_indices][:, :A]  # [num_valid, A]

    # Step 4: 批量计算每个团的边权重之和
    # 使用高级索引一次性提取所有团的子图
    # valid_cliques: [num_valid, A]
    # 我们需要计算每个团内所有边的权重之和

    # 方法：对每个团，提取其在SC2中的子矩阵，然后对上三角求和
    # 创建所有边对的索引
    edge_pairs = []
    for j in range(A):
        for k in range(j + 1, A):
            edge_pairs.append((j, k))
    edge_pairs = torch.tensor(edge_pairs, dtype=torch.long, device=device)  # [num_edges, 2]

    # 批量提取边权重
    # valid_cliques[:, edge_pairs[:, 0]]: [num_valid, num_edges] - 所有边的起点
    # valid_cliques[:, edge_pairs[:, 1]]: [num_valid, num_edges] - 所有边的终点
    src_indices = valid_cliques[:, edge_pairs[:, 0]]  # [num_valid, num_edges]
    dst_indices = valid_cliques[:, edge_pairs[:, 1]]  # [num_valid, num_edges]

    # 批量获取边权重: SC2_search[src, dst]
    # 需要展平索引来使用高级索引
    num_edges = edge_pairs.size(0)
    src_flat = src_indices.reshape(-1)  # [num_valid * num_edges]
    dst_flat = dst_indices.reshape(-1)  # [num_valid * num_edges]

    edge_weights_flat = SC2_search[src_flat, dst_flat]  # [num_valid * num_edges]
    edge_weights_matrix = edge_weights_flat.reshape(num_valid, num_edges)  # [num_valid, num_edges]

    # 对每个团的所有边求和
    edge_weights = edge_weights_matrix.sum(dim=1)  # [num_valid]

    # Step 5: 按边权重排序，返回top-X个团
    num_to_return = min(X, num_valid)
    sorted_weights, sorted_indices = torch.topk(edge_weights, num_to_return, largest=True)

    top_cliques = valid_cliques[sorted_indices]  # [num_to_return, A]
    top_sizes = torch.full((num_to_return,), A, dtype=torch.long, device=device)

    return top_cliques, top_sizes


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
