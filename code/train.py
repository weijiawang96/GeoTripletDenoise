"""Train GeoTripletDenoise in two stages (left:right triplet weight = 2:1)."""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.patch import PairedPatchDataset
from network.network import GeoTripletDenoise
from models.losses import estimate_curvature_from_fake, triplet_loss
from models.negative_mining import select_negative
from utils.checkpoints import validate_stage1_checkpoint
from utils.runtime import Progress, seed_all, atomic_json
from utils.paths import PUNET_ROOT, OUTPUT_ROOT, writable_output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(PUNET_ROOT))
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--save_dir", default=str(OUTPUT_ROOT / "train/checkpoints")
    )
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--memory_limit_mb', type=int, default=0)
    # Fixed method parameters.
    parser.set_defaults(
        encode_knn=16,
        bank_size=1024,
        curv_thresh=0.05,
        curv_fallback_thresh=0.10,
        margin=0.3,
        margin_scale=0.1,
        right_triplet_weight=0.5,
        repuls_factor=0.0005,
        lr=1e-4,
    )
    args = parser.parse_args()
    if args.stage == 2 and not Path(args.encoder_ckpt).is_file():
        parser.error("Stage 2 requires --encoder_ckpt pointing to stage1_latest.pth.")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs >= 1 and batch_size >= 2 are required.")
    if args.batch_size > args.bank_size:
        parser.error("batch_size must not exceed the 1024-slot bank capacity.")
    try:
        args.save_dir = str(writable_output(args.save_dir))
    except ValueError as error:
        parser.error(str(error))
    if (Path(args.save_dir) / f"stage{args.stage}_latest.pth").exists():
        parser.error(
            "This stage already has weights in save_dir; choose a new run directory."
        )
    return args


def main():
    args = parse_args()
    if args.memory_limit_mb:
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_mb / total_mb, 0)
    seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader = DataLoader(
        PairedPatchDataset(args.dataset_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    model = GeoTripletDenoise(encode_knn=args.encode_knn).cuda()
    if args.stage == 2:
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
        validate_stage1_checkpoint(
            ckpt, bank_size=args.bank_size, seed=args.seed, encode_knn=args.encode_knn
        )
        model.load_state_dict(ckpt["state_dict"], strict=True)
        opt_params = [
            p
            for n, p in model.named_parameters()
            if "left_proj_head" not in n and "right_proj_head" not in n
        ]
    else:
        opt_params = [
            p
            for n, p in model.named_parameters()
            if "conv_g" not in n
            and "attn" not in n
            and ("conv_f" not in n)
            and ("mlp_out" not in n)
        ]
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120, 160], gamma=0.1
    )
    bank_ptr = 0
    bank_filled = 0
    bank_left_feats = F.normalize(torch.randn(args.bank_size, 128), p=2, dim=1).cuda()
    bank_right_feats = F.normalize(torch.randn(args.bank_size, 128), p=2, dim=1).cuda()
    # Keep reserved draws in this order to preserve the seeded training sequence.
    _reserved_left_feats = F.normalize(
        torch.randn(args.bank_size, 128), p=2, dim=1
    ).cuda()
    _reserved_right_feats = F.normalize(
        torch.randn(args.bank_size, 128), p=2, dim=1
    ).cuda()
    bank_curvs = torch.rand(args.bank_size).cuda()
    bank_ids = torch.zeros(args.bank_size, dtype=torch.long).cuda() - 1
    config = {
        "format_version": 1,
        "model": "GeoTripletDenoise",
        "stage": args.stage,
        "stage1_positive": "once_noisy",
        "negative_mining": "curv_hard",
        "negative_candidates": "filled_entries",
        "encode_knn": args.encode_knn,
        "bank_size": args.bank_size,
        "curv_thresh": args.curv_thresh,
        "curv_fallback_thresh": args.curv_fallback_thresh,
        "margin": args.margin,
        "margin_scale": args.margin_scale,
        "right_triplet_weight": args.right_triplet_weight,
        "repuls_factor": args.repuls_factor,
        "stage1_loss": "(left_triplet + right_triplet_weight * right_triplet) / (1 + right_triplet_weight)",
        "stage2_loss": "mse + repuls_factor * rep",
        "stage2_policy": "joint_finetune_except_left_right_projection_heads",
        "checkpoint_policy": "latest_only_each_epoch",
        "environment": "score-denoise",
        "seed": args.seed,
    }
    print("GeoTripletDenoise configuration:", config, flush=True)
    for epoch in range(args.epochs):
        model.train()
        if args.stage == 2:
            model.left_proj_head.eval()
            model.right_proj_head.eval()
        pbar = Progress(train_loader, stage=args.stage, epoch=epoch + 1,
                        epochs=args.epochs, every=args.log_every, save_dir=args.save_dir)
        for data in pbar:
            optimizer.zero_grad(set_to_none=True)
            p_noisier = data["noisier_pat_pts"].float().cuda()
            p_fake = data["fakeclean_pat_pts"].float().cuda()
            p_positive = data["stage1_pos_pat_pts"].float().cuda()
            noisy_ctr = data["noisy_pat_ctr"].float().cuda()
            shape_ids = data["shape_id"].cuda()
            if args.stage == 1:
                filled_before_update = bank_filled
                candidate_source = 'filled_bank' if bank_filled else 'current_batch'
                f_right_a = model.right_triplet_feat(p_noisier)
                f_right_p = model.right_triplet_feat(p_positive)
                curv_a = estimate_curvature_from_fake(p_fake)
                f_right_n, right_sel = select_negative(
                    f_right_a,
                    bank_right_feats,
                    bank_curvs,
                    bank_ids,
                    curv_a,
                    shape_ids,
                    args.curv_thresh,
                    args.curv_fallback_thresh,
                    filled=bank_filled,
                )
                right_loss, right_stats = triplet_loss(
                    f_right_a,
                    f_right_p,
                    f_right_n,
                    margin=args.margin,
                    margin_scale=args.margin_scale,
                )
                f_left_a = model.left_triplet_feat(p_noisier)
                f_left_p = model.left_triplet_feat(p_positive)
                f_left_n, left_sel = select_negative(
                    f_left_a,
                    bank_left_feats,
                    bank_curvs,
                    bank_ids,
                    curv_a,
                    shape_ids,
                    args.curv_thresh,
                    args.curv_fallback_thresh,
                    filled=bank_filled,
                )
                left_loss, left_stats = triplet_loss(
                    f_left_a,
                    f_left_p,
                    f_left_n,
                    margin=args.margin,
                    margin_scale=args.margin_scale,
                )
                loss = (left_loss + args.right_triplet_weight * right_loss) / (
                    1.0 + args.right_triplet_weight
                )
                with torch.no_grad():
                    idx = (
                        torch.arange(
                            bank_ptr, bank_ptr + f_right_a.size(0), device="cuda"
                        )
                        % args.bank_size
                    )
                    bank_left_feats[idx] = f_left_a.detach()
                    bank_right_feats[idx] = f_right_a.detach()
                    bank_curvs[idx] = curv_a.detach()
                    bank_ids[idx] = shape_ids.detach()
                    bank_ptr = (bank_ptr + f_right_a.size(0)) % args.bank_size
                    bank_filled = min(args.bank_size, bank_filled + f_right_a.size(0))
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.6f}",
                        "Ltri": f"{left_stats['loss'].item():.4f}",
                        "Rtri": f"{right_stats['loss'].item():.4f}",
                        "Lsel": f"{left_sel.mean().item():.4f}",
                        "Rsel": f"{right_sel.mean().item():.4f}",
                        "Cdiff": f"{((right_sel + left_sel) * 0.5).mean().item():.4f}",
                        "filled": bank_filled,
                        "filled_before_update": filled_before_update,
                        "candidate_source": candidate_source,
                    }
                )
            else:
                pred, _ = model(p_noisier)
                mse_loss = torch.mean(torch.sum((pred - noisy_ctr * 2) ** 2, dim=1))
                rep_loss = torch.mean(
                    torch.max((p_fake - pred.unsqueeze(1)).pow(2).sum(dim=-1), dim=1)[0]
                )
                loss = mse_loss + args.repuls_factor * rep_loss
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.6f}",
                        "MSE": f"{mse_loss.item():.6f}",
                        "Rep": f"{rep_loss.item():.6f}",
                    }
                )
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
        scheduler.step()
        ckpt = {
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
        }
        path = Path(args.save_dir) / f"stage{args.stage}_latest.pth"
        temporary = path.with_suffix('.pth.tmp')
        torch.save(ckpt, temporary)
        os.replace(temporary, path)
        atomic_json(Path(args.save_dir).parent / f'stage{args.stage}_checkpoint.json',
                    {'epoch': epoch + 1, 'path': str(path), 'updated_unix': __import__('time').time()})
        print(f'CHECKPOINT stage={args.stage} epoch={epoch + 1} path={path}', flush=True)


if __name__ == "__main__":
    main()
