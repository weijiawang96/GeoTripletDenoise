"""Original PUNet inference protocol for the selected latest checkpoint."""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from datasets.pcl import PointcloudPatchDataset
from network.network import GeoTripletDenoise

from utils.paths import DEFAULT_CHECKPOINT, PUNET_ROOT, writable_output
from utils.runtime import seed_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--testset_root", type=str, default=str(PUNET_ROOT / "examples/PUNet"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--noise_lvls", type=float, default=0.01)
    parser.add_argument("--denoise_iters", type=int, default=-1)
    parser.add_argument("--encode_knn", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if not Path(args.ckpt_path).is_file():
        parser.error("Checkpoint not found. Use checkpoints/stage2_latest.pth or pass --ckpt_path.")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers nonnegative")
    try:
        writable_output(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    if args.denoise_iters < 0:
        args.denoise_iters = 2 if abs(args.noise_lvls - 0.03) < 1e-12 else 1

    model = GeoTripletDenoise(encode_knn=args.encode_knn).cuda()
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    with open(f"{args.testset_root}_shapes.txt", "r") as f:
        shapes = [x.strip() for x in f.readlines() if x.strip()]

    for res in ["10000_poisson", "50000_poisson"]:
        testset_dir = f"{args.testset_root}_{res}_{args.noise_lvls}"
        save_dir = os.path.join(args.output_dir, f"{res}_{args.noise_lvls}")
        os.makedirs(save_dir, exist_ok=True)

        for shapename in shapes:
            orig = np.loadtxt(os.path.join(testset_dir, f"{shapename}.xyz"))
            np.savetxt(
                os.path.join(save_dir, f"{shapename}_pred_iter_0.xyz"),
                orig.astype("float32"),
            )

            for i in range(args.denoise_iters):
                dset = PointcloudPatchDataset(
                    root=save_dir,
                    shape_name=f"{shapename}_pred_iter_{i}",
                    patch_radius=0.05,
                    points_per_patch=500,
                    state="testing",
                )
                loader = torch.utils.data.DataLoader(
                    dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

                pred_pts = []
                with torch.no_grad():
                    for patch, inv, disp in loader:
                        pred, _ = model(patch.float().cuda())
                        pred = torch.bmm(inv.float().cuda(), pred.unsqueeze(2)).squeeze(
                            2
                        )
                        pred_pts.append(
                            pred.cpu().numpy() * dset.patch_radius_absolute
                            + disp.numpy()
                        )

                np.savetxt(
                    os.path.join(save_dir, f"{shapename}_pred_iter_{i + 1}.xyz"),
                    np.vstack(pred_pts).astype("float32"),
                )


if __name__ == "__main__":
    main()
