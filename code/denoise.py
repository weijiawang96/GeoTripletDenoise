"""Denoise a single XYZ file with the pretrained GeoTripletDenoise network."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from network.network import GeoTripletDenoise
from datasets.pcl import PointcloudPatchDataset

from utils.paths import DEFAULT_CHECKPOINT, writable_output
from utils.runtime import seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ckpt_path",
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--denoise_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    seed_all(args.seed)
    if not Path(args.ckpt_path).is_file():
        parser.error("Checkpoint not found. Use checkpoints/stage2_latest.pth or pass --ckpt_path.")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers nonnegative")
    if args.denoise_iters < 1:
        parser.error("--denoise_iters must be positive")
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    try:
        writable_output(target)
    except ValueError as error:
        parser.error(str(error))
    if target.suffix != ".xyz":
        parser.error("--output must end with .xyz")
    if source == target:
        parser.error("Choose an output different from the input")
    points = np.loadtxt(source)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        parser.error(
            "Input must contain at least 3 finite XYZ rows (exactly 3 columns)"
        )
    model = GeoTripletDenoise().cuda()
    model.load_state_dict(
        torch.load(args.ckpt_path, map_location="cpu")["state_dict"], strict=True
    )
    model.eval()
    target.parent.mkdir(parents=True, exist_ok=True)
    # The original PUNet entry point writes and reloads float32 XYZ at each pass.
    np.savetxt(target, points.astype("float32"))
    for _ in range(args.denoise_iters):
        dset = PointcloudPatchDataset(
            root=str(target.parent),
            shape_name=target.stem,
            patch_radius=0.05,
            points_per_patch=500,
            state="testing",
        )
        loader = DataLoader(
            dset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        predicted = []
        with torch.no_grad():
            for patch, inv, displacement in loader:
                pred, _ = model(patch.float().cuda())
                pred = torch.bmm(inv.float().cuda(), pred.unsqueeze(2)).squeeze(2)
                predicted.append(
                    pred.cpu().numpy() * dset.patch_radius_absolute
                    + displacement.numpy()
                )
        np.savetxt(target, np.vstack(predicted).astype("float32"))


if __name__ == "__main__":
    main()
