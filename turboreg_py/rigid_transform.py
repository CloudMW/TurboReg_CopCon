"""
Rigid Transform Module
Provides rigid transformation (rotation + translation) computation
"""

import torch
from typing import Optional


class RigidTransform:
    """Rigid transformation class (SE(3))"""

    def __init__(self, transformation: torch.Tensor):
        """
        Initialize RigidTransform

        Args:
            transformation: 4x4 transformation matrix
        """
        assert transformation.shape == (4, 4), "Transformation must be 4x4 matrix"
        self.transformation = transformation

    def get_transformation(self) -> torch.Tensor:
        """Get the transformation matrix"""
        return self.transformation

    def get_rotation(self) -> torch.Tensor:
        """Get the rotation matrix (3x3)"""
        return self.transformation[:3, :3]

    def get_translation(self) -> torch.Tensor:
        """Get the translation vector (3x1)"""
        return self.transformation[:3, 3:4]

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to points

        Args:
            points: Points to transform [N, 3]

        Returns:
            Transformed points [N, 3]
        """
        R = self.get_rotation()
        t = self.get_translation()
        return points @ R.T + t.T


def integrate_trans(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Integrate rotation matrix R and translation vector t into 4x4 transformation matrix

    Args:
        R: Rotation matrices [bs, 3, 3]
        t: Translation vectors [bs, 3, 1]

    Returns:
        Transformation matrices [bs, 4, 4]
    """
    assert R.size(1) == 3 and R.size(2) == 3, "R must be [bs, 3, 3]"
    assert t.size(1) == 3 and t.size(2) == 1, "t must be [bs, 3, 1]"

    bs = R.size(0)
    trans = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(bs, 1, 1)

    trans[:, :3, :3] = R
    trans[:, :3, 3] = t.view(-1, 3)

    return trans


def rigid_transform_3d(
    A: torch.Tensor,
    B: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    weight_threshold: float = 0.0
) -> torch.Tensor:
    """
    Compute rigid transformation from point set A to point set B
    Solves: B = R @ A + t

    Args:
        A: Source point set [bs, N, 3]
        B: Target point set [bs, N, 3]
        weights: Optional weights for each point [bs, N]
        weight_threshold: Threshold for weight filtering

    Returns:
        Transformation matrices [bs, 4, 4]
    """
    bs = A.size(0)  # Batch size

    # Initialize weights if not provided
    if weights is None or weights.numel() == 0:
        W = torch.ones_like(A[:, :, 0])  # [bs, N]
    else:
        W = weights.clone()

    # Apply weight threshold
    W = W.masked_fill(W < weight_threshold, 0)

    # Compute weighted centroids
    centroid_A = (A * W.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
                 (W.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # [bs, 1, 3]
    centroid_B = (B * W.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
                 (W.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # [bs, 1, 3]

    # Center the point sets
    Am = A - centroid_A  # [bs, N, 3]
    Bm = B - centroid_B  # [bs, N, 3]

    # Compute cross-covariance matrix H
    H = torch.bmm(Am.permute(0, 2, 1), Bm * W.unsqueeze(-1))  # [bs, 3, 3]

    # SVD to get rotation
    U, S, Vt = torch.svd(H)

    # Ensure proper rotation (det(R) = 1)
    delta_UV = torch.det(Vt.bmm(U.permute(0, 2, 1)))
    eye = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(bs, 1, 1)
    eye[:, 2, 2] = delta_UV
    R = Vt.bmm(eye).bmm(U.permute(0, 2, 1))  # [bs, 3, 3]

    # Compute translation
    t = centroid_B.permute(0, 2, 1) - R.bmm(centroid_A.permute(0, 2, 1))  # [bs, 3, 1]

    # Return 4x4 transformation matrix
    return integrate_trans(R, t)

