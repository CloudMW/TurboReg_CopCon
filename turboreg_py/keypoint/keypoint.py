import torch.nn.functional as F
import torch


def detection_scores( neighbor, features):
    # neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]
    # first_pcd_length, second_pcd_length = inputs['stack_lengths'][0]
    neighbor = neighbor
    pcd_length = features.shape[0]
    # first_pcd_indices = torch.arange(first_pcd_length)
    # second_pcd_indices = torch.arange(first_pcd_length, first_pcd_length+second_pcd_length)

    # add a fake point in the last row for shadow neighbors
    shadow_features = torch.zeros_like(features[:1, :])
    features = torch.cat([features, shadow_features], dim=0)
    shadow_neighbor = torch.ones_like(neighbor[:1, :]) * (pcd_length)
    neighbor = torch.cat([neighbor, shadow_neighbor], dim=0)

    # #  normalize the feature to avoid overflow
    # point_cloud_feature0 = torch.max(features[first_pcd_indices])
    # point_cloud_feature1 = torch.max(features[second_pcd_indices])
    # max_per_sample =  torch.cat([
    #     torch.stack([point_cloud_feature0] * first_pcd_length, dim=0),
    #     torch.stack([point_cloud_feature1] * (second_pcd_length+1), dim=0)
    # ], dim=0)
    features = features / (torch.max(features) + 1e-6)

    # local max score (saliency score)
    neighbor_features = features[neighbor, :]  # [n_points, n_neighbors, 64]
    neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
    neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True)  # [n_points, 1]
    neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, 64]
    local_max_score = F.softplus(features - mean_features)  # [n_points, 64]

    # calculate the depth-wise max score
    depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
    depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]

    all_scores = local_max_score * depth_wise_max_score
    # use the max score among channel to be the score of a single point.
    scores = torch.max(all_scores, dim=1, keepdims=True)[0]  # [n_points, 1]

    # hard selection (used during test)
    # if self.training is False:

    # local_max = torch.max(neighbor_features, dim=1)[0]
    # is_local_max = (features == local_max)
    # # print(f"Local Max Num: {float(is_local_max.sum().detach().cpu())}")
    # detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
    # scores = scores * detected

    return scores[:-1, :]


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


def visual_by_score(points, scores, point_size: float = 2.0, show: bool = True):
    """
    Visualize points with colors determined by scores.
    Scores are ranked descending -> color maps from red (high score) to white (low score).

    Args:
        points: torch.Tensor or array-like, shape [N,3]
        scores: torch.Tensor or array-like, shape [N] or [N,1]
        point_size: float, size used for Open3D render option
        show: bool, if True open an Open3D window; if False just return the point cloud data

    Returns:
        If Open3D available: (pcd, colors_numpy) where pcd is an open3d.geometry.PointCloud
        Otherwise: (points_numpy, colors_numpy)
    """
    # Convert inputs
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    device = points.device
    points_np = points.detach().cpu().numpy()

    scores_flat = scores.detach().cpu().view(-1).float()
    N = points_np.shape[0]

    # If scores length doesn't match points, try to broadcast or raise
    if scores_flat.shape[0] != N:
        raise ValueError(f"scores length ({scores_flat.shape[0]}) does not match points ({N})")

    # Convert scores to rank-based normalized values in [0,1]
    if N == 1:
        norm = torch.tensor([1.0])
    else:
        # Higher score -> closer to 1
        sorted_idx = torch.argsort(scores_flat, descending=True)
        ranks = torch.empty_like(sorted_idx)
        ranks[sorted_idx] = torch.arange(0, N, device=ranks.device)
        # rank 0 => highest score
        ranks = ranks.float()
        norm = 1.0 - (ranks / float(max(1, N - 1)))

    norm_np = norm.cpu().numpy()

    # Map normalized value to color: red (high score) -> blue (low score)
    # norm==1 -> red (1,0,0); norm==0 -> blue (0,0,1)
    colors = torch.stack([
        norm,           # R channel increases with score
        torch.zeros_like(norm),
        1.0 - norm      # B channel decreases with score
    ], dim=1)
    colors_np = colors.cpu().numpy()

    # Try to use Open3D for visualization; if unavailable, return arrays
    try:
        import open3d as o3d
    except Exception:
        print("Open3D not available; returning numpy arrays instead of visualizing.")
        return points_np, colors_np

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    if show:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='visual_by_score', width=800, height=600, visible=True)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        try:
            opt.point_size = float(point_size)
        except Exception:
            # older Open3D versions might require different handling; ignore if fails
            pass
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

    return pcd, colors_np


def get_keypoint_from_scores(points,features):

    neighbors_num = 5
    neighbors_idx = neighbors(points,neighbors_num)
    score = detection_scores(neighbors_idx,features)
    # _, topk_idx = torch.topk(score.squeeze(-1), k=k, dim=0, largest=True, sorted=True)
    return score
    # visual_by_score(points,score)