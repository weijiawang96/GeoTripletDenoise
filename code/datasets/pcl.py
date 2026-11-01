import os
import json
from pathlib import Path
import random
import torch
import numpy as np
import scipy.spatial as sp
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils.geometry import pca_alignment


class PointcloudPatchDataset(Dataset):

    def __init__(
        self,
        root=None,
        shapes_list_file=None,
        patch_radius=0.05,
        points_per_patch=500,
        seed=None,
        state="train",
        shape_name=None,
        pcl_noisy=None,
    ):

        self.root = root
        self.shapes_list_file = shapes_list_file

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.state = state

        # 初始化随机数生成器，用于 patch 内采样
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**10 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.noise_shapes = []

        self.shape_names = []
        # 测试阶段：从文件读取点云
        if self.state == "testing" and shape_name is not None:
            noise_pts = np.loadtxt(
                os.path.join(self.root, shape_name + ".xyz")
            )  # load the pcl file

        # 验证阶段：直接使用传入点云
        elif self.state == "validating":
            noise_pts = pcl_noisy.numpy()  # directly use the pass-in pcl

        # print(shape_name)
        # print(noise_pts.shape)
        # exit()
        noise_kdtree = sp.cKDTree(noise_pts)
        self.noise_shapes.append({"noise_pts": noise_pts, "noise_kdtree": noise_kdtree})
        self.shape_patch_count.append(noise_pts.shape[0])
        bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
        self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def patch_sampling(self, patch_pts):
        if patch_pts.shape[0] > self.points_per_patch:
            sample_index = np.random.choice(
                range(patch_pts.shape[0]), self.points_per_patch, replace=False
            )
        else:
            sample_index = np.random.choice(
                range(patch_pts.shape[0]), self.points_per_patch
            )
        return sample_index

    def __getitem__(self, index):
        # 根据全局索引定位该点所属的 shape

        shape_ind, patch_ind = self.shape_index(index)
        noise_shape = self.noise_shapes[shape_ind]
        patch_radius = self.patch_radius_absolute[shape_ind]

        # 查询 noise patch
        noise_patch_idx = noise_shape["noise_kdtree"].query_ball_point(
            noise_shape["noise_pts"][patch_ind], patch_radius
        )

        if len(noise_patch_idx) < 3:
            # 针对稀疏点做特殊处理：若不足 3 个点，重复索引补齐，保证 PCA 可用
            diff = 3 - len(noise_patch_idx)
            for i in range(diff):
                noise_patch_idx.append(random.choice(noise_patch_idx))

        noise_patch_pts = (
            noise_shape["noise_pts"][noise_patch_idx]
            - noise_shape["noise_pts"][patch_ind]
        )
        noise_patch_pts, noise_patch_inv = pca_alignment(noise_patch_pts)
        noise_patch_pts /= patch_radius

        noise_sample_idx = self.patch_sampling(noise_patch_pts)
        noise_patch_pts = noise_patch_pts[noise_sample_idx]

        # support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
        # support_radius = np.expand_dims(support_radius, axis=0)

        if self.state == "testing" or self.state == "validating":
            return (
                torch.from_numpy(noise_patch_pts),
                torch.from_numpy(noise_patch_inv),
                noise_shape["noise_pts"][patch_ind],
            )

    def __len__(self):
        return sum(self.shape_patch_count)

    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if (index >= shape_patch_offset) and (
                index < shape_patch_offset + shape_patch_count
            ):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind


class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, "pointclouds", split, resolution)
        self.transform = transform  # 一次增强流程
        self.pointclouds = []  # 仅存储点坐标
        self.normals = []  # 预留：用于存储法向

        self.pointcloud_names = []
        # Preserve the historical shape IDs across filesystem copies.
        order_file = Path(root) / dataset / "file_order.json"
        names = os.listdir(self.pcl_dir)
        if order_file.is_file():
            key = "pointclouds/%s/%s" % (split, resolution)
            recorded = json.loads(order_file.read_text())["directories"].get(key)
            if recorded is not None:
                if set(recorded) != {n for n in names if n.endswith(".xyz")}:
                    raise ValueError("Dataset files differ from file_order.json: " + key)
                names = recorded
        for fn in tqdm(names, desc="Loading"):
            if fn[-3:] != "xyz":
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError("File not found: %s" % pcl_path)
            pcl = torch.FloatTensor(
                np.loadtxt(pcl_path, dtype=np.float32)
            )  # 6 维（xyz + normals）
            # print(pcl)
            pcl_pts = pcl[:, :3]  # 点坐标
            # pcl_nmls = pcl[:, 3:]   # nmls

            self.pointclouds.append(pcl_pts)
            # self.normals.append(pcl_nmls)

            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        # print(len(self.pointclouds))
        # print(len(self.normals))
        # exit()
        # np.savetxt("./before_pcl_clean.xyz", self.pointclouds[idx])
        # np.savetxt("./before_pcl_w_nml_clean.xyz", np.hstack((self.pointclouds[idx], self.normals[idx])))

        # 这里以 clean 点云为基础样本
        data = {
            "pcl_clean": self.pointclouds[idx].clone(),
            # 'pcl_normals': self.normals[idx],  # I think no need to clone
            "name": self.pointcloud_names[idx],
        }

        if self.transform is not None:  # 在这里执行加噪等变换
            data = self.transform(data)

        return data
