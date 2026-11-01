"""Denoise the organized PCPNet, KinectFusion or RueMadame inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from utils.paths import DEFAULT_CHECKPOINT, CODE_ROOT, DATA_ROOT, OUTPUT_ROOT, writable_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["PCPNet", "KinectFusion", "StreetScenes"], required=True)
    parser.add_argument("--split", default="testset_low_noise", help="PCPNet official split stem")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir")
    parser.add_argument("--denoise_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--list_only", action="store_true")
    args = parser.parse_args()
    index = json.loads((DATA_ROOT / "index.json").read_text())
    if args.dataset == "PCPNet":
        if args.split not in index["PCPNet"]["splits"]:
            parser.error("Unknown PCPNet split: " + args.split)
        names = index["PCPNet"]["splits"][args.split]
        cases = [{"name": n, "input": "PCPNet/pointclouds/" + n + ".xyz"} for n in names]
    else:
        cases = index[args.dataset]["cases"]
    if args.list_only:
        print(json.dumps({"dataset": args.dataset, "cases": cases}, indent=2))
        return
    checkpoint = Path(args.ckpt_path).resolve()
    if not checkpoint.is_file():
        parser.error("Use checkpoints/stage2_latest.pth or pass --ckpt_path.")
    if args.denoise_iters < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("iterations and batch_size must be positive; num_workers must be nonnegative")
    try:
        output = writable_output(args.output_dir or OUTPUT_ROOT / (args.dataset + "_inference"))
    except ValueError as error:
        parser.error(str(error))
    if output.exists() and any(output.iterdir()):
        parser.error("output_dir is not empty; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    record = vars(args).copy()
    record.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  environment="score-denoise", cases=cases, status="RUNNING")
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        for case in cases:
            command = [sys.executable, "-u", str(CODE_ROOT / "denoise.py"),
                       "--input", str(DATA_ROOT / case["input"]),
                       "--output", str(output / (case["name"] + "_pred.xyz")),
                       "--ckpt_path", str(checkpoint), "--denoise_iters", str(args.denoise_iters),
                       "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
                       "--seed", str(args.seed)]
            print("Denoising " + case["name"], flush=True)
            subprocess.run(command, env=env, check=True)
    except Exception:
        record["status"] = "FAILED"
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        raise
    record["status"] = "COMPLETE"
    manifest.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
