"""
TurboReg GPU Implementation
Main registration class for point cloud alignment
"""

import torch
from .model_selection import ModelSelection, string_to_metric_type
from .rigid_transform import RigidTransform
from .core_turboreg import verification_v2_metric, post_refinement
from .utils_pcr import coplanar_constraint, coplanar_constraint_more_points
from turboreg_py.demo_py.utils_pcr import *
class TurboRegGPU:
    """
    TurboReg GPU accelerated point cloud registration
    Fast and robust registration using clique-based hypothesis generation
    """

    def __init__(
            self,
            max_N: int,
            tau_length_consis: float,
            num_pivot: int,
            radiu_nms: float,
            tau_inlier: float,
            metric_str: str = "IN"
    ):
        """
        Initialize TurboRegGPU

        Args:
            max_N: Maximum number of correspondences to use
            tau_length_consis: Length consistency threshold (τ)
            num_pivot: Number of pivot points (K_1)
            radiu_nms: Radius for NMS to avoid solution instability
            tau_inlier: Inlier threshold for post-refinement
            metric_str: Metric for model selection ("IN", "MAE", or "MSE")
        """
        self.max_N = max_N
        self.tau_length_consis = tau_length_consis
        self.num_pivot = num_pivot
        self.radiu_nms = radiu_nms
        self.tau_inlier = tau_inlier
        self.hard = True  # Hard compatibility graph
        self.eval_metric = string_to_metric_type(metric_str)

    def run_reg(
            self,
            corr_kpts_src: torch.Tensor, corr_kpts_dst: torch.Tensor, trans_gt: torch.Tensor, src_cloud: torch.Tensor,
            dst_cloud: torch.Tensor, kpts_src: torch.Tensor, kpts_dst: torch.Tensor, corr_ind: torch.Tensor,
            feature_kpts_src:torch.Tensor = None,
            feature_kpts_dst:torch.Tensor = None,
            # kpts_src: torch.Tensor,
            #     kpts_dst: torch.Tensor,
            #     pts_src: torch.Tensor,
            #     pts_dst: torch.Tensor,
            #     corr_ind: torch.Tensor
    ) -> torch.Tensor:
        """
        Run registration and return transformation matrix

        Args:
            kpts_src: Source keypoints [N, 3]
            kpts_dst: Target keypoints [N, 3]

        Returns:
            Transformation matrix [4, 4]
        """
        rigid_transform = self.run_reg_cxx(corr_kpts_src, corr_kpts_dst, trans_gt, src_cloud, dst_cloud, kpts_src,
                                           kpts_dst, corr_ind,feature_kpts_src = feature_kpts_src, feature_kpts_dst=feature_kpts_dst)
        return rigid_transform.get_transformation()

    def run_reg_cxx(
            self,
            corr_kpts_src: torch.Tensor, corr_kpts_dst: torch.Tensor, trans_gt: torch.Tensor, src_cloud: torch.Tensor,
            dst_cloud: torch.Tensor, kpts_src: torch.Tensor, kpts_dst: torch.Tensor, corr_ind: torch.Tensor,
            feature_kpts_src: torch.Tensor = None,
            feature_kpts_dst: torch.Tensor = None,
    ) -> RigidTransform:
        """
        Run registration and return RigidTransform object

        Args:
            kpts_src: Source keypoints [N, 3]
            kpts_dst: Target keypoints [N, 3]

        Returns:
            RigidTransform object
        """
        # Control the number of keypoints
        N_node = min(corr_kpts_src.size(0), self.max_N)
        if N_node < corr_kpts_src.size(0):
            corr_kpts_src = corr_kpts_src[:N_node]
            corr_kpts_dst = corr_kpts_dst[:N_node]
        k_cliques_size = 5
        # Compute C2 (compatibility matrix)
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
        if not self.hard:
            C2 = torch.relu(1 - (cross_dist / self.tau_length_consis) ** 2)
        else:
            C2 = (cross_dist < self.tau_length_consis).float()

        # Apply mask based on distance threshold (NMS)
        mask = (src_dist + target_dist) <= self.radiu_nms
        C2 = C2.masked_fill(mask, 0)

        # Compute SC2 (compatibility scores)
        # Align with C++: SC2 = (C2 @ C2) * C2 (Hadamard product with C2)
        SC2 = torch.matmul(C2, C2) * C2

        # Select pivots
        SC2_up = torch.triu(SC2, diagonal=1)  # Upper triangular
        flat_SC2_up = SC2_up.flatten()
        scores_topk, idx_topk = torch.topk(flat_SC2_up, self.num_pivot)

        # Convert flat indices to 2D indices
        pivots = torch.stack([
            (idx_topk // N_node).long(),
            (idx_topk % N_node).long()
        ], dim=1)  # [num_pivot, 2]

        # Find 3-cliques
        SC2_for_search = SC2_up.clone()

        SC2_pivot_0 = SC2_for_search[pivots[:, 0]] > 0  # [num_pivot, N]
        SC2_pivot_1 = SC2_for_search[pivots[:, 1]] > 0  # [num_pivot, N]
        indic_c3_torch = SC2_pivot_0 & SC2_pivot_1  # [num_pivot, N]

        SC2_pivots = SC2_for_search[pivots[:, 0], pivots[:, 1]]  # [num_pivot]

        # Calculate scores for each 3-clique
        SC2_ADD_C3 = (
                SC2_pivots.unsqueeze(1) +
                SC2_for_search[pivots[:, 0]] +
                SC2_for_search[pivots[:, 1]]
        )  # [num_pivot, N]

        # Mask the C3 scores
        SC2_C3 = SC2_ADD_C3 * indic_c3_torch.float()
        cliques_dyn = True
        if cliques_dyn:
            # Get top-2 indices for each row
            topk_K2 = torch.topk(SC2_C3, k=k_cliques_size-2, dim=1)[1]  # [num_pivot, 2]

            # Initialize cliques tensor [num_pivot*2, 3]
            num_pivots = pivots.size(0)
            cliques_tensor = torch.zeros(
                (num_pivots , k_cliques_size),
                dtype=torch.long,
                device=corr_kpts_src.device
            )

            # Upper part
            cliques_tensor[:num_pivots, :2] = pivots

            for idx in range(k_cliques_size-2):
                cliques_tensor[:num_pivots, 2 + idx] = topk_K2[:, idx]

        else:

            topk_K2 = torch.topk(SC2_C3, k=2, dim=1)[1]
            # Initialize cliques tensor [num_pivot*2, 3]

            num_pivots = pivots.size(0)
            cliques_tensor = torch.zeros(
                (num_pivots *2,3),
                dtype=torch.long,
                device=corr_kpts_src.device
            )

            # Upper part
            cliques_tensor[:num_pivots, :2] = pivots
            cliques_tensor[:num_pivots, 2] = topk_K2[:, 0]
            # Lower part
            cliques_tensor[num_pivots:, :2] = pivots
            cliques_tensor[num_pivots:, 2] = topk_K2[:, 1]

        # Apply coplanar constraint (align with C++ behavior)
        cliques_tensor = coplanar_constraint_more_points(
            cliques_tensor,
            corr_kpts_src,
            corr_kpts_dst,
            kpts_src,
            kpts_dst,
            corr_ind,

            k=100
        )

        # local filter
        # from turboreg_py.local_filter import local_filter
        # cliques_tensor = local_filter(
        #     cliques_tensor,
        #     corr_kpts_src,
        #     corr_kpts_dst,
        #     kpts_src,
        #     kpts_dst,
        #     corr_ind,
        #     feature_kpts_src = None,  # Disable feature-based filtering to avoid index bounds issues
        #     feature_kpts_dst = None,
        #     threshold=0.05,
        #     k=20,
        #     num_cliques=10
        # )

        # from turboreg_py.sphere_filter import sphere_filter
        # cliques_tensor = sphere_filter(
        #     cliques_tensor,
        #     corr_kpts_src,
        #     corr_kpts_dst,
        #     kpts_src,
        #     kpts_dst,
        #     corr_ind,
        #     k=10,
        # )


        # Verification with metric selection
        model_selector = ModelSelection(self.eval_metric, self.tau_inlier)
        best_in_num, best_trans, res, cliques_wise_trans, idx_best_guess = verification_v2_metric(
            cliques_tensor,
            corr_kpts_src,
            corr_kpts_dst,
            model_selector
        )




        # Post refinement
        refined_trans = post_refinement(
            best_trans,
            corr_kpts_src,
            corr_kpts_dst,
            it_num=20,
            inlier_threshold=self.tau_inlier
        )

        vis = True
        if vis:
            refined_trans_numpy = refined_trans.cpu().numpy()
            trans_gt_numpy = trans_gt.cpu().numpy()
            rre, rte =compute_transformation_error(trans_gt_numpy, refined_trans_numpy)
            is_succ = (rre < 15) & (rte < 0.3)
            if not is_succ:
                import turboreg_py.visualization_debug as vis_debug
                vis_debug.visualization_from_clique_configurable(src_cloud.cpu().numpy(), dst_cloud.cpu().numpy(),
                                                                 kpts_src.cpu().numpy(), kpts_dst.cpu().numpy(),
                                                                 corr_ind.cpu().numpy(), trans_gt.cpu().numpy(),
                                                                 best_trans.cpu().numpy(),
                                                                 set(cliques_tensor.cpu().numpy()[idx_best_guess]),
                                                                 )

        trans_final = RigidTransform(refined_trans)
        return trans_final
