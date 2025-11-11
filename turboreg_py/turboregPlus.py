"""
TurboReg GPU Implementation
Main registration class for point cloud alignment
"""
import torch

from .model_selection import ModelSelection, string_to_metric_type
from .rigid_transform import RigidTransform
from .core_turboreg import verification_v2_metric, post_refinement
from turboreg_py.demo_py.utils_pcr import *


class TurboRegPlus:
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
            feature_kpts_src: torch.Tensor = None,
            feature_kpts_dst: torch.Tensor = None,
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
                                           kpts_dst, corr_ind, feature_kpts_src=feature_kpts_src,
                                           feature_kpts_dst=feature_kpts_dst)
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

        # regor
        from turboreg_py.regor.regor import Regenerator
        regenerator = Regenerator()
        src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
            corr_kpts_src.unsqueeze(0),
            corr_kpts_dst.unsqueeze(0),
            kpts_src.unsqueeze(0),
            kpts_dst.unsqueeze(0),
            feature_kpts_src.unsqueeze(0),
            feature_kpts_dst.unsqueeze(0),
            trans_gt.unsqueeze(0),
            knn_num=100,
            sampling_num=100
        )

        src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
            src_keypts_corr_final,
            tgt_keypts_corr_final,
            kpts_src.unsqueeze(0),
            kpts_dst.unsqueeze(0),
            feature_kpts_src.unsqueeze(0),
            feature_kpts_dst.unsqueeze(0),
            trans_gt.unsqueeze(0),
            knn_num=20,
            sampling_num=500
        )


        labels_regor = self.inlier_ratio_by_point(src_keypts_corr_final[0], tgt_keypts_corr_final[0], corr_ind, trans_gt)
        inlier_ratio_regor = labels_regor.float().sum() / labels_regor.size(0)
        print(f'inlier_ratio regior: {inlier_ratio_regor.item():.4f} regor num : {src_keypts_corr_final.size(1)}')
        # # 计算 原点 和 目标 点 的重叠率

        """
        
        ## keypoints select
        from turboreg_py.keypoint.keypoint import get_keypoint_from_scores
        from turboreg_py.keypoint.keypoints_optimal_transport import optimal_transport
        from turboreg_py.keypoint.keypoint_optional_spatrl_graph import keypoint_spectral_graph
        src_keypoint_index,tgt_keypoint_index =  keypoint_spectral_graph(kpts_src, kpts_dst, feature_kpts_src, feature_kpts_dst)


        from turboreg_py.visualization.visualization_keypoint import visualize_keypoint
        # visualize_keypoint(kpts_src,kpts_dst,src_keypoint_index,tgt_keypoint_index,trans_gt)
         # = optimal_transport()
        kpts_src_old = kpts_src
        kpts_dst_old = kpts_dst
        kpts_src = kpts_src[src_keypoint_index]
        kpts_dst = kpts_dst[tgt_keypoint_index]


        # 计算 原点 和 目标 点 的重叠率
        src_overlap_ratio_plus, _ = self.get_overlap_ratio(kpts_src, kpts_dst_old, trans_gt)
        _, dst_overlap_ratio_plus = self.get_overlap_ratio(kpts_src_old, kpts_dst, trans_gt)
        print(f"src_overlap plus : {src_overlap_ratio_plus} points {kpts_src.shape[0]}, dst_overlap plus: {dst_overlap_ratio_plus} points {kpts_dst.shape[0]}")




        src_keypoint_feature = feature_kpts_src[src_keypoint_index]
        tgt_keypoint_feature = feature_kpts_dst[tgt_keypoint_index]
        # corr_ind = self.get_corr_k_num(feature_kpts_src, tgt_keypoint_feature)
        corr_ind = self.get_corr_k_num(feature_kpts_src, tgt_keypoint_feature,True,5)
        labels = self.inlier_ratio(kpts_src_old, kpts_dst, corr_ind, trans_gt)
        inlier_ratio = labels.float().sum() / labels.size(0)
        print(f'inlier_ratio src to tgt: {inlier_ratio.item():.4f}')
        corr_ind[:,1]=tgt_keypoint_index[corr_ind[:,1]]

        # corr_ind = self.get_corr_k_num(feature_kpts_src, tgt_keypoint_feature)
        corr_ind_tgt_to_src = self.get_corr_k_num(feature_kpts_dst, src_keypoint_feature,True,5)
        labels = self.inlier_ratio(kpts_src, kpts_dst_old, corr_ind_tgt_to_src[:,[1,0]], trans_gt)
        inlier_ratio = labels.float().sum() / labels.size(0)
        print(f'inlier_ratio tgt to src: {inlier_ratio.item():.4f}')
        corr_ind_tgt_to_src[:,1] = src_keypoint_index[corr_ind_tgt_to_src[:,1]]
        corr_ind_tgt_to_src = corr_ind_tgt_to_src[:,[1,0]]
        corr_ind =  torch.concat((corr_ind, corr_ind_tgt_to_src)).unique(dim=0)
        labels = self.inlier_ratio(kpts_src_old, kpts_dst_old, corr_ind, trans_gt)
        inlier_ratio = labels.float().sum() / labels.size(0)
        print(f'inlier_ratio all: {inlier_ratio.item():.4f}')
        # src_keypoint_feature = feature_kpts_src[src_keypoint_index]
        # tgt_keypoint_feature = feature_kpts_dst[tgt_keypoint_index]
        # corr_ind = self.get_corr_k_num(src_keypoint_feature, feature_kpts_dst,True,5)
        # labels = self.inlier_ratio(kpts_src, kpts_dst_old, corr_ind, trans_gt)
        # inlier_ratio = labels.float().sum() / labels.size(0)
        # print(f'inlier_ratio 1tok: {inlier_ratio.item():.4f}')

"""

        # corr_kpts_src = kpts_src_old[corr_ind[:, 0]]
        # corr_kpts_dst = kpts_dst_old[corr_ind[:, 1]]
        corr_kpts_src =src_keypts_corr_final[0]
        corr_kpts_dst = tgt_keypts_corr_final[0]

        # N_node = min(corr_kpts_src.size(0), self.max_N)
        # if N_node < corr_kpts_src.size(0):
        #     corr_kpts_src = corr_kpts_src[:N_node]
        #     corr_kpts_dst = corr_kpts_dst[:N_node]
        N_node = min(corr_kpts_src.size(0), self.max_N)
        if N_node < corr_kpts_src.size(0):
            corr_kpts_src = corr_kpts_src[:N_node]
            corr_kpts_dst = corr_kpts_dst[:N_node]
        k_cliques_size = 3
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

        # from turboreg_py.seed_point import cal_leading_eigenvector,pick_seeds
        # SC_dist_thre = 0.1
        # SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        # confidence = cal_leading_eigenvector(SC_measure.unsqueeze(0), method='power')
        # seeds = pick_seeds(src_dist.unsqueeze(0), confidence, R=self.radiu_nms, max_num=int(N_node * 0.2)).squeeze(0)
        #
        # SC2 = SC2[seeds][:,seeds]

        ##Select pivots
        N_node = SC2.size(0)
        SC2_up = torch.triu(SC2, diagonal=1)  # Upper triangular
        flat_SC2_up = SC2_up.flatten()
        scores_topk, idx_topk = torch.topk(flat_SC2_up,  min(self.num_pivot//3, corr_kpts_src.size(0) * (corr_kpts_src.size(0) - 1) // 2))

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

        topk_K2 = torch.topk(SC2_C3, k=2, dim=1)[1]
        # Initialize cliques tensor [num_pivot*2, 3]

        num_pivots = pivots.size(0)
        cliques_tensor = torch.zeros(
            (num_pivots * 2, 3),
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

        # cliques_tensor =seeds[cliques_tensor]
        # cliques_tensor = coplanar_constraint_more_points(
        #     cliques_tensor,
        #     corr_kpts_src,
        #     corr_kpts_dst,
        #     kpts_src,
        #     kpts_dst,
        #     corr_ind,
        #     k=200
        # )

        # visualize_point_cloud_with_voxel_grid_lines(src_cloud)
        # cliques_tensor = coplanar_constraint(
        #     cliques_tensor,
        #     corr_kpts_src,
        #     corr_kpts_dst,
        #     kpts_src,
        #     kpts_dst,
        #     corr_ind,
        #     k=500
        # )

        # local filter
        # from turboreg_py.local_filter import local_filter,local_filter_2
        # cliques_tensor = local_filter_2(
        #     cliques_tensor,
        #     corr_kpts_src,
        #     corr_kpts_dst,
        #     kpts_src,
        #     kpts_dst,
        #     src_cloud,
        #     dst_cloud,
        #     corr_ind,
        #     trans_gt = trans_gt,
        #     feature_kpts_src = feature_kpts_src,  # Disable feature-based filtering to avoid index bounds issues
        #     feature_kpts_dst = feature_kpts_dst,
        #     threshold=0.01,
        #     num_cliques=20
        # )
        #

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

        vis = False
        if vis:
            refined_trans_numpy = refined_trans.cpu().numpy()
            trans_gt_numpy = trans_gt.cpu().numpy()
            rre, rte = compute_transformation_error(trans_gt_numpy, refined_trans_numpy)
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

    def inlier_ratio(self, src_keypts, tgt_keypts, corr, gt_trans):
        def transform(pts, trans):
            """
            Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
            Input
                - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
                - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
            Output
                - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
            """
            if len(pts.shape) == 3:
                trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
                return trans_pts.permute(0, 2, 1)
            else:
                trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
                return trans_pts.T

        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        # distance = torch.sqrt(torch.sum(torch.power(frag1_warp - frag2, 2), axis=1))
        distance = torch.norm(frag1_warp - frag2, dim=1)
        labels = (distance < self.tau_inlier)
        return labels
    def inlier_ratio_by_point(self, src_keypts, tgt_keypts, corr, gt_trans):
        def transform(pts, trans):
            """
            Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
            Input
                - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
                - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
            Output
                - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
            """
            if len(pts.shape) == 3:
                trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
                return trans_pts.permute(0, 2, 1)
            else:
                trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
                return trans_pts.T

        # build the ground truth label
        frag1 = src_keypts
        frag2 = tgt_keypts
        frag1_warp = transform(frag1, gt_trans)
        # distance = torch.sqrt(torch.sum(torch.power(frag1_warp - frag2, 2), axis=1))
        distance = torch.norm(frag1_warp - frag2, dim=1)
        labels = (distance < self.tau_inlier)
        return labels

    def get_corr(self, src_desc, tgt_desc):
        distance = torch.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = torch.argmin(distance, axis=1)
        use_mutual = True
        if use_mutual:
            target_idx = torch.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == torch.arange(source_idx.shape[0]).to(source_idx.device))
            corr = torch.concatenate(
                [torch.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
                axis=-1)
        else:
            corr = torch.concatenate([torch.arange(source_idx.shape[0]).to(source_idx.device)[:, None], source_idx[:, None]], axis=-1)

        return corr
    def get_overlap_ratio(self, src_keypts, tgt_keypts,gt_trans):
        def transform(pts, trans):
            """
            Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
            Input
                - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
                - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
            Output
                - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
            """
            if len(pts.shape) == 3:
                trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
                return trans_pts.permute(0, 2, 1)
            else:
                trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
                return trans_pts.T

        # build the ground truth label
        frag1 = src_keypts
        frag2 = tgt_keypts
        frag1_warp = transform(frag1, gt_trans)
        # distance = torch.sqrt(torch.sum(torch.power(frag1_warp - frag2, 2), axis=1))
        frag1_warp_expanded = frag1_warp.unsqueeze(1)
        frag2_expanded = frag2.unsqueeze(0)
        diff = frag1_warp_expanded - frag2_expanded  # [M, N, 3]
        dist_matrix = torch.norm(diff, p=2, dim=2)
        dis_src_2_dst = dist_matrix.min(dim=-1)[0]
        dis_dst_2_src = dist_matrix.min(dim=0)[0]
        src_overlap_ratio = (dis_src_2_dst<self.tau_inlier).sum()/(dist_matrix.shape[0])
        dst_overlap_ratio = (dis_dst_2_src<self.tau_inlier).sum()/(dist_matrix.shape[-1])
        return src_overlap_ratio, dst_overlap_ratio
    # def get_corr_k_num(self,src_desc:torch.Tensor, tgt_desc:torch.Tensor,k=5):
    #     device = src_desc.device
    #     dot_product = torch.matmul(src_desc, tgt_desc.T)  # [N, M]：源描述子与目标描述子的点积
    #     distance = torch.sqrt(2 - 2 * dot_product + 1e-6)  # [N, M]：欧式距离，加1e-6避免sqrt(0)
    #
    #     # 3. 对每个源特征点，按距离升序排序，取前k个目标特征点的索引
    #     _, source_idx = torch.sort(distance, dim=1)  # [N, M]：排序后的目标索引（升序，越小越近）
    #     source_idx = source_idx[:, :k]  # [N, k]：取前k个近邻的目标索引
    #     source_idx = source_idx.flatten()  # [N*k,]：展平为1维（对齐原numpy的flatten()）
    #     use_mutual =True
    #     # 4. 生成匹配对（corr）：分use_mutual=True/False两种情况
    #     if use_mutual:
    #         # 互近邻筛选：找到“源→目标是近邻，且目标→源也是近邻”的匹配对
    #         _, target_idx = torch.sort(distance, dim=0)  # [M,]：每个目标特征点的最近源特征点索引（axis=0对应源维度）
    #         target_idx = target_idx[:,:k].flatten()
    #         # 验证：源→目标的近邻（source_idx）是否也是目标→源的近邻
    #         # target_idx[source_idx]：每个“源→目标”对应的“目标→源”索引
    #         # np.arange(source_idx.shape[0]) → PyTorch对应torch.arange(source_idx.numel(), device=device)
    #         mutual_nearest = (target_idx[source_idx] == torch.arange(source_idx.numel(), device=device))
    #
    #         # 提取互近邻的匹配对：[源索引, 目标索引]
    #         # 原numpy的np.where(mutual_nearest == 1)[0] → PyTorch的torch.nonzero(mutual_nearest, as_tuple=True)[0]
    #         src_corr_idx = torch.nonzero(mutual_nearest, as_tuple=True)[0]  # 源侧匹配索引（对应source_idx的位置）
    #         tgt_corr_idx = source_idx[mutual_nearest]  # 目标侧匹配索引
    #
    #         # 拼接为[N, 2]的匹配对矩阵
    #         corr = torch.cat([
    #             src_corr_idx.unsqueeze(1),  # [K, 1]
    #             tgt_corr_idx.unsqueeze(1)  # [K, 1]
    #         ], dim=1)
    #
    #     else:
    #         # 不筛选：每个源特征点对应k个目标近邻，生成[N*k, 2]的匹配对
    #         # 原numpy的np.repeat(np.arange(src_keypts.shape[0])[:, None], k, axis=0)
    #         src_corr_idx = torch.arange(src_desc.shape[0], device=device)  # [N,]：源特征点索引
    #         src_corr_idx = src_corr_idx.unsqueeze(1).repeat(1, k).flatten()  # [N*k,]：每个源索引重复k次
    #
    #         # 拼接为[N*k, 2]的匹配对矩阵
    #         corr = torch.cat([
    #             src_corr_idx.unsqueeze(1),  # [N*k, 1]
    #             source_idx.unsqueeze(1)  # [N*k, 1]
    #         ], dim=1)
    #
    #     # 可选：转回numpy（若需要和原代码输出格式一致）
    #     # corr = corr.cpu().numpy()
    #
    #     return corr

    def get_corr_k_num(self, src_desc: torch.Tensor, tgt_desc: torch.Tensor, use_mutual: bool = True, k: int = 5):
        """
        Args:
            src_desc: [M, F]
            tgt_desc: [N, F]
            use_mutual: 是否使用互相匹配策略
            k: 当 use_mutual=True 时，检查目标的前 k 个最近源点中是否包含源点 J

        Returns:
            corr: [K, 2] (K == M when use_mutual=False；当 use_mutual=True 时 K<=M)
                  每行为 [src_idx, tgt_idx]
        """
        device = src_desc.device
        M = src_desc.shape[0]
        N = tgt_desc.shape[0]
        eps = 1e-8

        # 直接用点积构造距离：distance = sqrt(2 - 2 * dot)（适用于归一化描述子）
        dot = torch.matmul(src_desc, tgt_desc.t())  # [M, N]
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=eps))  # [M, N]

        # 对每个源，找到最近的目标索引 i_j
        nearest_tgt = torch.argmin(dist, dim=1)  # [M]

        if not use_mutual:
            src_idx = torch.arange(M, device=device)
            corr = torch.stack([src_idx, nearest_tgt], dim=1)  # [M, 2]
            return corr

        # use_mutual == True
        # 限制 k 不超过源点数
        k = max(1, min(k, M))

        # 对每个目标（按列）找到按距离升序的源点索引
        _, src_sorted_per_tgt = torch.sort(dist, dim=0)  # [M, N]
        topk_src_per_tgt = src_sorted_per_tgt[:k, :]  # [k, N]

        # 构建一个布尔矩阵 topk_mask[s, t] 表示源 s 是否在目标 t 的前 k 个最近源中
        topk_rows = topk_src_per_tgt.reshape(-1)  # [k*N]
        topk_cols = torch.arange(N, device=device).unsqueeze(0).repeat(k, 1).reshape(-1)  # [k*N]
        topk_mask = torch.zeros((M, N), dtype=torch.bool, device=device)
        topk_mask[topk_rows, topk_cols] = True

        # 对每个源 j，检查其最近目标 i_j 是否把 j 包含在该目标的前 k 源中
        src_indices = torch.arange(M, device=device)
        mutual_mask = topk_mask[src_indices, nearest_tgt]  # [M] bool

        # 筛选满足互近邻条件的配对
        matched_src = src_indices[mutual_mask]
        matched_tgt = nearest_tgt[mutual_mask]
        if matched_src.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        corr = torch.stack([matched_src, matched_tgt], dim=1)  # [K, 2]

        return corr