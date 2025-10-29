"""
Model Selection Module
Provides different metrics for selecting the best transformation hypothesis
"""

import torch
from enum import Enum
from typing import Union


class MetricType(Enum):
    """Metric types for model selection"""
    INLIER_COUNT = "IN"
    MAE = "MAE"
    MSE = "MSE"


def string_to_metric_type(metric_str: str) -> MetricType:
    """Convert string to MetricType enum"""
    metric_map = {
        "IN": MetricType.INLIER_COUNT,
        "MAE": MetricType.MAE,
        "MSE": MetricType.MSE
    }

    if metric_str not in metric_map:
        raise ValueError(f"Invalid metric type string: {metric_str}")

    return metric_map[metric_str]


class ModelSelection:
    """
    Model selection for point cloud registration
    Selects the best transformation based on different metrics
    """

    def __init__(self, metric: Union[str, MetricType], inlier_threshold: float):
        """
        Initialize ModelSelection

        Args:
            metric: Metric type (IN, MAE, MSE) or MetricType enum
            inlier_threshold: Threshold for determining inliers
        """
        if isinstance(metric, str):
            self.metric_type = string_to_metric_type(metric)
        else:
            self.metric_type = metric

        self.inlier_threshold = inlier_threshold

    def calculate_best_clique(
        self,
        cliques_wise_trans: torch.Tensor,
        kpts_src: torch.Tensor,
        kpts_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the best matching clique based on the given metric

        Args:
            cliques_wise_trans: Transformation matrices for each clique [C, 4, 4]
            kpts_src: Source keypoints [N, 3]
            kpts_dst: Target keypoints [N, 3]

        Returns:
            Index of the best clique
        """
        # Extract rotation and translation
        cliques_wise_trans_3x3 = cliques_wise_trans[:, :3, :3]  # [C, 3, 3]
        cliques_wise_trans_3x1 = cliques_wise_trans[:, :3, 3:4]  # [C, 3, 1]

        # Transform source keypoints: R @ kpts_src.T + t
        kpts_src_prime = torch.einsum('cnm,mk->cnk', cliques_wise_trans_3x3, kpts_src.T) + cliques_wise_trans_3x1
        kpts_src_prime = kpts_src_prime.permute(0, 2, 1)  # [C, N, 3]

        # Calculate residuals
        res = torch.norm(kpts_src_prime - kpts_dst.unsqueeze(0), p=2, dim=-1)  # [C, N]
        indic_in = res < self.inlier_threshold  # Inlier indicators

        # Count inliers for each clique
        cliquewise_in_num = indic_in.sum(dim=-1).float()  # [C]

        # Select best clique based on metric
        if self.metric_type == MetricType.INLIER_COUNT:
            idx_best_guess = cliquewise_in_num.argmax()

        elif self.metric_type == MetricType.MAE:
            # MAE with weighted inliers
            mae_weights = (self.inlier_threshold - res).clamp(min=0) / self.inlier_threshold
            idx_best_guess = (indic_in.float() * mae_weights).sum(dim=-1).argmax()

        elif self.metric_type == MetricType.MSE:
            # MSE with weighted inliers
            mse_weights = ((self.inlier_threshold - res).clamp(min=0) ** 2) / (self.inlier_threshold ** 2)
            idx_best_guess = (indic_in.float() * mse_weights).sum(dim=-1).argmax()

        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

        return idx_best_guess

