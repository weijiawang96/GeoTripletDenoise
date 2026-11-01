import random
import torch
from torchvision.transforms import Compose


class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2  # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (
                (pcl**2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]
            )  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert (
            "pcl_noisy" not in data
        ), "Point clouds must be normalized before applying noise perturbation."
        data["pcl_clean"], center, scale = self.normalize(data["pcl_clean"])
        data["center"] = center
        data["scale"] = scale

        return data


class AddNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):

        # Orig ver, single add noise pass ------------------------------------------------------------
        """
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data['pcl_noisy'] = data['pcl_clean'] + torch.randn_like(data['pcl_clean']) * noise_std
        data['noise_std'] = noise_std
        """

        # Noisier2Noise ver, add same noise level -----------------------------------------------------
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        additional_noise = torch.randn_like(data["pcl_clean"]) * noise_std
        data["pcl_noisy"] = data["pcl_clean"] + additional_noise

        more_noise = (
            torch.randn_like(data["pcl_clean"]) * noise_std
        )  # 8.28 TODO: different noise
        data["pcl_noisier"] = (
            data["pcl_noisy"] + more_noise
        )  # add the same noise to pcl_noisy

        data["noisier2noise_diff"] = more_noise  # noise diff = noisier pcl - noisy pcl
        data["noise_std"] = noise_std

        # TEMP: Save the noisy and clean clouds as xyz.---
        # np.savetxt("./pcl_noisier.xyz", data['pcl_noisier'])
        # np.savetxt("./pcl_noisy.xyz", data['pcl_noisy'])
        # np.savetxt("./pcl_clean.xyz", data['pcl_clean'])
        # exit()
        # -------------------------------------------------

        return data


def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=False):
    transforms = [
        NormalizeUnitSphere(),  # Must normalise to unit sphere before adding noise
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        # RandomScale([1.0-scale_d, 1.0+scale_d]),
    ]
    # No Rotation
    return Compose(transforms)
