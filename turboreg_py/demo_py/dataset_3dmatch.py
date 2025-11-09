import os
import numpy as np
import open3d as o3d


class TDMatchFCGFAndFPFHDataset:
    def __init__(self, base_dir, dataset_type="3DMatch", descriptor_type="fcgf"):
        """
        Initialize the MatchDataset.

        :param base_dir: Base directory containing the datasets.
        :param dataset_type: Type of the dataset ("3DMatch" or "3DLoMatch").
        :param descriptor_type: Descriptor type ("fcgf" or "fpfh").
        """
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.descriptor_type = descriptor_type
        self.dataset_path = os.path.join(
            base_dir, dataset_type, f"all_{descriptor_type}")
        self.inlier_threshold = 0.1
        if not os.path.exists(self.dataset_path):
            raise ValueError(
                f"Dataset path {self.dataset_path} does not exist.")

        self.matching_pairs = self._load_matching_pairs()

    def _load_matching_pairs(self):
        """
        Load all matching pairs from the dataset.

        :return: List of matching pair information.
        """
        matching_pairs = []

        # Traverse all scenes in the dataset
        # for scene_dir in tqdm(os.listdir(self.dataset_path), desc="Loading scenes"):
        for scene_dir in sorted(os.listdir(self.dataset_path)):
            scene_path = os.path.join(self.dataset_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            # Collect all matching files in the scene
            target_gt_name = "@corr_fcgf.txt" if self.descriptor_type == "fcgf" else "@corr.txt"
            for file_name in sorted(os.listdir(scene_path)):
                if file_name.endswith(target_gt_name):
                    base_name = file_name.split("@")[0]
                    if self.descriptor_type == "fpfh":
                        corr_file = os.path.join(scene_path, file_name)
                        gtmat_file = os.path.join(
                            scene_path, f"{base_name}@GTmat.txt")
                        corr_ind_file = os.path.join(scene_path, f"{base_name}@corr_ind.txt")
                        src_npz_file = os.path.join(scene_path, f"{base_name.split('+')[0]}_fpfh.npz")
                        dst_npz_file = os.path.join(scene_path, f"{base_name.split('+')[1]}_fpfh.npz")
                        label_file = os.path.join(scene_path, f"{base_name}@label.txt")
                    elif self.descriptor_type == "fcgf":
                        corr_file = os.path.join(scene_path, file_name)
                        gtmat_file = os.path.join(
                            scene_path, f"{base_name}@GTmat_{self.descriptor_type}.txt")
                        corr_ind_file = os.path.join(scene_path, f"{base_name}@corr_ind_{self.descriptor_type}.txt")
                        src_npz_file = os.path.join(scene_path, f"{base_name.split('+')[0]}_fcgf.npz")
                        dst_npz_file = os.path.join(scene_path, f"{base_name.split('+')[1]}_fcgf.npz")
                        label_file = os.path.join(scene_path, f"{base_name}@label_{self.descriptor_type}.txt")
                    src_cloud_file = os.path.join(
                        scene_path, f"{base_name.split('+')[0]}.ply")
                    dst_cloud_file = os.path.join(
                        scene_path, f"{base_name.split('+')[1]}.ply")
                    src_kpts_file = os.path.join(scene_path, f"{base_name.split('+')[0]}.pcd")
                    dst_kpts_file = os.path.join(scene_path, f"{base_name.split('+')[1]}.pcd")
                    if os.path.exists(gtmat_file) and os.path.exists(src_cloud_file) and os.path.exists(dst_cloud_file):
                        matching_pairs.append({
                            "corr_file": corr_file,
                            "label_file": label_file,
                            "gtmat_file": gtmat_file,
                            "src_cloud_file": src_cloud_file,
                            "dst_cloud_file": dst_cloud_file,
                            "corr_ind_file":corr_ind_file,
                            "src_npz_file": src_npz_file,
                            "dst_npz_file": dst_npz_file,
                            "src_kpts_file": src_kpts_file,
                            "dst_kpts_file": dst_kpts_file,
                        })

        return matching_pairs

    def __len__(self):
        """
        Return the number of matching pairs in the dataset.

        :return: Number of matching pairs.
        """
        return len(self.matching_pairs)

    def __getitem__(self, idx):
        """
        Get a matching pair by index.

        :param idx: Index of the matching pair.
        :return: Dictionary containing kpts_src, kpts_dst, transformation matrix, and point clouds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        pair_info = self.matching_pairs[idx]

        # Load correspondences
        correspondences = np.loadtxt(pair_info["corr_file"], delimiter=' ')
        corr_kpts_src = correspondences[:, :3]
        corr_kpts_dst = correspondences[:, 3:]

        # Load ground truth transformation matrix
        trans_gt = np.loadtxt(pair_info["gtmat_file"], delimiter=' ')

        # Load point clouds
        src_cloud = np.asarray(o3d.io.read_point_cloud(
            pair_info["src_cloud_file"]).points)
        dst_cloud = np.asarray(o3d.io.read_point_cloud(
            pair_info["dst_cloud_file"]).points)

        corr_ind = np.loadtxt(pair_info["corr_ind_file"], dtype=np.int64)

        kpts_src = np.asarray(o3d.io.read_point_cloud(
            pair_info["src_kpts_file"]).points)
        kpts_dst = np.asarray(o3d.io.read_point_cloud(
            pair_info["dst_kpts_file"]).points)

        label = np.loadtxt(pair_info["label_file"], dtype=np.int32)
        if os.path.exists(pair_info["src_npz_file"]) is False:

            feature_kpts_src = None
            feature_kpts_dst = None
        else :
            src_npz = np.load(pair_info["src_npz_file"])
            dst_npz = np.load(pair_info["dst_npz_file"])



            feature_kpts_src = src_npz['feature']
            feature_kpts_dst = dst_npz['feature']

            cal_corr,cal_lable = self.get_corr_k_num(pair_info["src_npz_file"], pair_info["dst_npz_file"], trans_gt)
        return {
            "corr_kpts_src": corr_kpts_src,
            "corr_kpts_dst": corr_kpts_dst,
            "trans_gt": trans_gt,
            "pts_src": src_cloud,
            "pts_dst": dst_cloud,
            "kpts_src": kpts_src,
            "kpts_dst": kpts_dst,
            "corr_ind": corr_ind,
            "feature_kpts_src": feature_kpts_src,
            "feature_kpts_dst": feature_kpts_dst,
        }


    def get_corr(self,src_npz_file,tgt_npz_file,gt_trans):
        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor_type == 'fcgf':
            src_data = np.load(src_npz_file)
            tgt_data = np.load(tgt_npz_file)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor_type == 'fpfh':
            src_data = np.load(src_npz_file)
            tgt_data = np.load(tgt_npz_file)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        # orig_trans = np.linalg.inv(self.gt_trans[key])  # the given ground truth trans is target-> source
        # # data augmentation
        # aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        # aug_T = translation_matrix(self.augment_translation)
        # aug_trans = integrate_trans(aug_R, aug_T)
        # tgt_keypts = transform(tgt_keypts, aug_trans)
        # gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        # if self.num_node == 'all':
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        # else:
        #     src_sel_ind = np.random.choice(N_src, self.num_node)
        #     tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        use_mutual = True
        if use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
                                  axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int32)

        return corr ,labels
    def get_corr_k_num(self,src_npz_file,tgt_npz_file,gt_trans,k=5):
        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor_type == 'fcgf':
            src_data = np.load(src_npz_file)
            tgt_data = np.load(tgt_npz_file)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor_type == 'fpfh':
            src_data = np.load(src_npz_file)
            tgt_data = np.load(tgt_npz_file)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        # orig_trans = np.linalg.inv(self.gt_trans[key])  # the given ground truth trans is target-> source
        # # data augmentation
        # aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        # aug_T = translation_matrix(self.augment_translation)
        # aug_trans = integrate_trans(aug_R, aug_T)
        # tgt_keypts = transform(tgt_keypts, aug_trans)
        # gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        # if self.num_node == 'all':
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        # else:
        #     src_sel_ind = np.random.choice(N_src, self.num_node)
        #     tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx =np.argsort(distance, axis=1)[:,:k].flatten()
        use_mutual = False
        if use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
                                  axis=-1)
        else:
            corr = np.concatenate([np.repeat(np.arange(src_keypts.shape[0])[:, None], repeats=k, axis=0), source_idx[:, None]], axis=-1)

        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int32)

        return corr ,labels
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
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T