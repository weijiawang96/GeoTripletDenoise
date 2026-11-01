import random
import numpy as np
from utils.compat import ball_query
import torch
from torch.utils.data import Dataset
from .pcl import PointCloudDataset
from utils.geometry import pca_alignment
from utils.transforms import standard_train_transforms


def _cutoff_from_idx(idx_row):
    invalid = (idx_row == -1).nonzero(as_tuple=True)[0]
    if invalid.numel() == 0:
        return idx_row.numel()
    return int(invalid[0])


def _ball_patch(center, pcl, k, radius):
    _, idx, pts = ball_query(center, pcl.unsqueeze(0), K=k, radius=radius)
    cutoff = _cutoff_from_idx(idx[0][0])
    if cutoff < 3:
        _, idx, pts = ball_query(
            center, pcl.unsqueeze(0), K=k, radius=radius * 1.5
        )
        cutoff = _cutoff_from_idx(idx[0][0])
        assert 3 <= cutoff <= k
    return pts[0][0][:cutoff]


def _transform_patch(patch_pts, origin, rot_inv, query_radius):
    patch_pts = patch_pts - origin
    patch_pts = torch.t(
        torch.from_numpy(np.linalg.inv(rot_inv) * np.matrix(patch_pts.T))
    )
    return patch_pts / query_radius


def make_patches_with_noisy_positive(
    noisier_pcl, noisy_pcl, fake_clean_pcl, patch_radius_ratio, num_patches
):
    """Return anchor, fake-clean target, and once-noisy positive patches in the same PCA frame."""
    noisier_shape_bbdiag = torch.linalg.norm(
        torch.max(noisier_pcl, dim=0)[0] - torch.min(noisier_pcl, dim=0)[0],
        2,
    )
    query_radius = noisier_shape_bbdiag * patch_radius_ratio

    num_points = noisier_pcl.size(0)
    seed_idx = torch.randperm(num_points)[:num_patches]
    seed_pnt_noisier = noisier_pcl[seed_idx].unsqueeze(0)
    seed_pnt_noisy = noisy_pcl[seed_idx].unsqueeze(0)
    empirical_cutoff_thresh = num_points // 10

    noisier_pat_pts = _ball_patch(
        seed_pnt_noisier,
        noisier_pcl,
        empirical_cutoff_thresh,
        query_radius,
    )
    fakeclean_pat_pts = _ball_patch(
        seed_pnt_noisy,
        fake_clean_pcl,
        empirical_cutoff_thresh,
        query_radius,
    )
    noisypos_pat_pts = _ball_patch(
        seed_pnt_noisy,
        noisy_pcl,
        empirical_cutoff_thresh,
        query_radius,
    )

    seed_noisier = seed_pnt_noisier[0][0]
    seed_noisy = seed_pnt_noisy[0][0]

    noisier_pat_pts = noisier_pat_pts - seed_noisier
    noisier_pat_pts, noisier_pat_rot_inv = pca_alignment(noisier_pat_pts)
    noisier_pat_pts = noisier_pat_pts / query_radius

    fakeclean_pat_pts = _transform_patch(
        fakeclean_pat_pts,
        seed_noisier,
        noisier_pat_rot_inv,
        query_radius,
    )
    noisypos_pat_pts = _transform_patch(
        noisypos_pat_pts,
        seed_noisier,
        noisier_pat_rot_inv,
        query_radius,
    )

    seed_noisy = seed_noisy - seed_noisier
    seed_noisy = seed_noisy.unsqueeze(0)
    seed_noisy = torch.t(
        torch.from_numpy(np.linalg.inv(noisier_pat_rot_inv) * np.matrix(seed_noisy.T))
    )
    seed_noisy = seed_noisy / query_radius

    return {
        "noisier_pat_pts": noisier_pat_pts,
        "fakeclean_pat_pts": fakeclean_pat_pts,
        "stage1_pos_pat_pts": noisypos_pat_pts,
        "noisy_pat_ctr": seed_noisy.squeeze(0),
    }


class PairedPatchDataset(Dataset):
    def __init__(self, root, resolutions=None, patch_size=500):
        super().__init__()
        if resolutions is None:
            resolutions = ["10000_poisson", "30000_poisson", "50000_poisson"]
        self.patch_size = patch_size
        self.patch_radius = 0.05
        self.datasets = [
            PointCloudDataset(
                root=root,
                dataset="",
                split="train",
                resolution=resl,
                transform=standard_train_transforms(
                    noise_std_max=0.020 / 1.414,
                    noise_std_min=0.005 / 1.414,
                ),
            )
            for resl in resolutions
        ]
        self.len_datasets = sum(len(dset) for dset in self.datasets)

    def __len__(self):
        return self.len_datasets * 1000

    def patch_sampling(self, length):
        return np.random.choice(
            range(length), self.patch_size, replace=(length < self.patch_size)
        )

    def __getitem__(self, idx):
        pcl_dset_idx = random.randint(0, len(self.datasets) - 1)
        shape_idx = idx % len(self.datasets[pcl_dset_idx])
        pcl_data = self.datasets[pcl_dset_idx][shape_idx]

        noisier_pcl = pcl_data["pcl_noisier"]
        noisy_pcl = pcl_data["pcl_noisy"]
        fake_clean_pcl = noisier_pcl - 2 * pcl_data["noisier2noise_diff"]

        data_dict = make_patches_with_noisy_positive(
            noisier_pcl,
            noisy_pcl,
            fake_clean_pcl,
            self.patch_radius,
            1,
        )
        return {
            "noisier_pat_pts": data_dict["noisier_pat_pts"][
                self.patch_sampling(data_dict["noisier_pat_pts"].shape[0])
            ],
            "fakeclean_pat_pts": data_dict["fakeclean_pat_pts"][
                self.patch_sampling(data_dict["fakeclean_pat_pts"].shape[0])
            ],
            "stage1_pos_pat_pts": data_dict["stage1_pos_pat_pts"][
                self.patch_sampling(data_dict["stage1_pos_pat_pts"].shape[0])
            ],
            "noisy_pat_ctr": data_dict["noisy_pat_ctr"],
            "shape_id": pcl_dset_idx * 10000 + shape_idx,
        }
