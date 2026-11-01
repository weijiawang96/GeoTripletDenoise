"""Run the selected two-stage experiment and save its output under outputs/."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from utils.paths import CODE_ROOT, PUNET_ROOT, OUTPUT_ROOT, DEFAULT_CHECKPOINT, writable_output

ROOT = CODE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(PUNET_ROOT))
    parser.add_argument(
        "--test_root", default=str(PUNET_ROOT / "examples/PUNet"), help="PUNet input prefix"
    )
    parser.add_argument(
        "--run_dir", default=str(OUTPUT_ROOT / "train")
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage1_epochs", type=int, default=200)
    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--ckpt_path", help="With --skip_train; defaults to checkpoints/stage2_latest.pth"
    )
    args = parser.parse_args()
    if args.ckpt_path and not args.skip_train:
        parser.error("--ckpt_path is only used with --skip_train")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        run_dir = writable_output(args.run_dir)
    except ValueError as error:
        parser.error(str(error))
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error("run_dir is not empty; choose a fresh directory to keep prior results.")
    ckpts = run_dir / "checkpoints"
    console = run_dir / "console"
    console.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(phase, script, *parts):
        command = [sys.executable, "-u", str(ROOT / script), *map(str, parts)]
        path = console / (phase + ".log")
        commands.append({"phase": phase, "argv": command, "log": str(path)})
        (run_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"Running {phase}; log: {path}", flush=True)
        with path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

    if not args.skip_train:
        if any((ckpts / f"stage{s}_latest.pth").exists() for s in (1, 2)):
            parser.error("run_dir already contains weights; choose a fresh directory")
        common = [
            "--dataset_root",
            args.data_root,
            "--save_dir",
            ckpts,
            "--batch_size",
            args.batch_size,
            "--num_workers",
            args.num_workers,
            "--seed", args.seed,
        ]
        run("stage1", "train.py", "--stage", 1, "--epochs", args.stage1_epochs, *common)
        run(
            "stage2",
            "train.py",
            "--stage",
            2,
            "--epochs",
            args.stage2_epochs,
            "--encoder_ckpt",
            ckpts / "stage1_latest.pth",
            *common,
        )
        selected = ckpts / "stage2_latest.pth"
    else:
        selected = (
            Path(args.ckpt_path)
            if args.ckpt_path
            else DEFAULT_CHECKPOINT
        )
    if not args.skip_test:
        if not selected.is_file():
            parser.error("Selected checkpoint does not exist: " + str(selected))
        for noise, iterations in [(0.01, 1), (0.02, 1), (0.03, 2)]:
            run(
                "test_" + str(noise),
                "test_punet.py",
                "--ckpt_path",
                selected,
                "--testset_root",
                args.test_root,
                "--output_dir",
                run_dir / "results",
                "--noise_lvls",
                noise,
                "--denoise_iters",
                iterations,
                "--batch_size",
                args.test_batch_size,
                "--num_workers",
                args.num_workers,
                "--seed", args.seed,
            )
        if not args.skip_eval:
            run(
                "evaluate",
                "evaluate.py",
                "--results_root",
                run_dir / "results",
                "--dataset_root",
                args.data_root,
            )


if __name__ == "__main__":
    main()
