import torch
def cal_leading_eigenvector( M, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(10):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)
def pick_seeds(dists, scores, R, max_num):
    """
    Select seeding points using Non Maximum Suppression. (here we only support bs=1)
    Input:
        - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
        - scores:      [bs, num_corr]     initial confidence of each correspondence
        - R:           float              radius of nms
        - max_num:     int                maximum number of returned seeds
    Output:
        - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
    """
    assert scores.shape[0] == 1

    # parallel Non Maximum Suppression (more efficient)
    score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
    # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
    score_relation = score_relation.bool() | (dists[0] >= R).bool()
    is_local_max = score_relation.min(-1)[0].float()

    score_local_max = scores * is_local_max
    sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

    # max_num = scores.shape[1]

    return_idx = sorted_score[:, 0: max_num].detach()

    return score_local_max



def get_seed(corr_kpts_src, corr_kpts_dst, hard=False,tau_length_consis=0.012):
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

    # Compute SC2 (compatibility scores)
    # Align with C++: SC2 = (C2 @ C2) * C2 (Hadamard product with C2)
    SC2 = torch.matmul(C2, C2) * C2


    SC_dist_thre = 0.1
    SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
    confidence = cal_leading_eigenvector(SC_measure.unsqueeze(0), method='power')

    score = pick_seeds(src_dist.unsqueeze(0), confidence, R=radiu_nms, max_num=int(M * 0.5)).squeeze(0)

    return score

