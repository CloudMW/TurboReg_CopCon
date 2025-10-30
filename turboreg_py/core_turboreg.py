"""
Core TurboReg GPU Module
Contains verification and post-refinement functions
"""

import torch
from typing import Tuple
from .rigid_transform import rigid_transform_3d
from .model_selection import ModelSelection


def transform(src_keypts: torch.Tensor, initial_trans: torch.Tensor) -> torch.Tensor:
    """
    Apply transformation to source keypoints
    
    Args:
        src_keypts: Source keypoints [N, 3]
        initial_trans: Transformation matrix [4, 4]
    
    Returns:
        Transformed keypoints [N, 3]
    """
    R = initial_trans[:3, :3]  # [3, 3]
    t = initial_trans[:3, 3:4]  # [3, 1]
    
    # R @ pts.T + t -> (N, 3)
    return src_keypts @ R.T + t.T


def verification(
    cliques_tensor: torch.Tensor,
    kpts_src: torch.Tensor,
    kpts_dst: torch.Tensor,
    inlier_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Verification function for finding best transformation
    
    Args:
        cliques_tensor: Clique indices [C, 3]
        kpts_src: Source keypoints [N, 3]
        kpts_dst: Target keypoints [N, 3]
        inlier_threshold: Threshold for inlier detection
    
    Returns:
        best_in_num: Number of inliers for best transformation
        best_in_indic: Inlier indicators for best transformation
        best_trans: Best transformation matrix
        res: Residuals for all cliques
        cliques_wise_trans: Transformations for all cliques
        cliquewise_in_num: Number of inliers for each clique
    """
    # Select keypoints for each clique
    kpts_src_sub = kpts_src[cliques_tensor.view(-1)].view(-1, 3, 3)  # [C, 3, 3]
    kpts_dst_sub = kpts_dst[cliques_tensor.view(-1)].view(-1, 3, 3)  # [C, 3, 3]
    
    # Compute transformation for each clique
    cliques_wise_trans = rigid_transform_3d(kpts_src_sub, kpts_dst_sub)  # [C, 4, 4]
    
    # Extract R and t
    cliques_wise_trans_3x3 = cliques_wise_trans[:, :3, :3]  # [C, 3, 3]
    cliques_wise_trans_3x1 = cliques_wise_trans[:, :3, 3:4]  # [C, 3, 1]
    
    # Transform source keypoints
    kpts_src_prime = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
    kpts_src_prime = kpts_src_prime.permute(0, 2, 1)  # [C, N, 3]
    
    # Calculate residuals
    res = torch.norm(kpts_src_prime - kpts_dst.unsqueeze(0), p=2, dim=-1)  # [C, N]
    indic_in = res < inlier_threshold  # [C, N]
    
    # Count inliers for each clique
    cliquewise_in_num = indic_in.sum(dim=-1).float()  # [C]
    
    # Find best clique (most inliers)
    idx_best_guess = cliquewise_in_num.argmax()
    
    best_in_num = cliquewise_in_num[idx_best_guess]
    best_trans = cliques_wise_trans[idx_best_guess]
    best_in_indic = indic_in[idx_best_guess]
    
    return best_in_num, best_in_indic, best_trans, res, cliques_wise_trans, cliquewise_in_num


def verification_v2_metric(
    cliques_tensor: torch.Tensor,
    kpts_src: torch.Tensor,
    kpts_dst: torch.Tensor,
    model_selector: ModelSelection
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]:
    """
    Verification function with custom metric selection
    
    Args:
        cliques_tensor: Clique indices [C, 3]
        kpts_src: Source keypoints [N, 3]
        kpts_dst: Target keypoints [N, 3]
        model_selector: Model selection object with metric
    
    Returns:
        best_in_num: Number of inliers (placeholder)
        best_trans: Best transformation matrix
        res: Residuals (placeholder)
        cliques_wise_trans: Transformations for all cliques
    """
    # Select keypoints for each clique
    N,C = cliques_tensor.shape
    kpts_src_sub = kpts_src[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]
    kpts_dst_sub = kpts_dst[cliques_tensor.view(-1)].view(-1, C, 3)  # [C, 3, 3]
    
    # Compute transformation for each clique
    cliques_wise_trans = rigid_transform_3d(kpts_src_sub, kpts_dst_sub)  # [C, 4, 4]
    
    # Use model selector to find best clique
    idx_best_guess = model_selector.calculate_best_clique(cliques_wise_trans, kpts_src, kpts_dst)
    
    best_trans = cliques_wise_trans[idx_best_guess]
    
    # Placeholders (not used in this version)
    best_in_num = torch.tensor(0.0)
    res = torch.tensor(0.0)
    
    return best_in_num, best_trans, res, cliques_wise_trans,idx_best_guess


def post_refinement(
    initial_trans: torch.Tensor,
    src_keypts: torch.Tensor,
    tgt_keypts: torch.Tensor,
    it_num: int,
    inlier_threshold: float = 0.1
) -> torch.Tensor:
    """
    Post-refinement using iterative reweighted least squares
    
    Args:
        initial_trans: Initial transformation matrix [4, 4]
        src_keypts: Source keypoints [N, 3]
        tgt_keypts: Target keypoints [N, 3]
        it_num: Number of iterations
        inlier_threshold: Threshold for inlier detection
    
    Returns:
        Refined transformation matrix [4, 4]
    """
    # Match C++ version: use dtype and device from initial_trans
    inlier_threshold_list = torch.full((it_num,), inlier_threshold,
                                       dtype=initial_trans.dtype,
                                       device=initial_trans.device)

    previous_inlier_num = 0
    pred_inlier = None
    
    for i in range(it_num):
        # Apply transformation
        warped_src_keypts = transform(src_keypts, initial_trans)
        
        # Calculate L2 distance
        L2_dis = torch.norm(warped_src_keypts - tgt_keypts, p=2, dim=1)  # [N]
        
        # Predict inliers
        pred_inlier = L2_dis < inlier_threshold_list[i]
        
        # Count inliers
        inlier_num = pred_inlier.sum().item()
        
        # Check convergence
        if abs(inlier_num - previous_inlier_num) < 1:
            break
        else:
            previous_inlier_num = inlier_num
        
        # Match C++ version: removed "if inlier_num == 0: break" check
        # This ensures consistent behavior with C++ implementation

        # Compute weights for inliers
        weight = 1 / (1 + (L2_dis[pred_inlier] / inlier_threshold_list[i]).pow(2))
        
        # Recompute transformation with weighted inliers
        initial_trans = rigid_transform_3d(
            src_keypts[pred_inlier].unsqueeze(0),
            tgt_keypts[pred_inlier].unsqueeze(0),
            weight.unsqueeze(0)
        ).squeeze(0)
    
    return initial_trans

