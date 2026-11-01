"""Verify organized data, strict checkpoint loading and bank behavior."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import hashlib

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import torch
import torch.nn.functional as F
from models.negative_mining import select_negative
from models.losses import select_hard_negative
from network.network import GeoTripletDenoise
from utils.paths import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, DEFAULT_CHECKPOINT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_train():
    spec = importlib.util.spec_from_file_location("project_training", CODE / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_negative_candidates():
    anchors = torch.zeros(4, 128)
    anchors[:, 0] = 1
    bank = anchors[:1].repeat(1024, 1)
    bank[:128] *= -1
    curves, ids = torch.zeros(1024), torch.full((1024,), -1, dtype=torch.long)
    ids[:128] = 1
    shape_ids, curvature = torch.zeros(4, dtype=torch.long), torch.zeros(4)
    arguments = (anchors, bank, curves, ids, curvature, shape_ids, .05, .1)
    assert torch.equal(select_negative(*arguments, filled=128)[0], -anchors)
    real = F.normalize(torch.randn(4, 128), dim=1)
    shape_ids = torch.arange(4)
    cold = select_negative(real, bank, curves, ids, curvature, shape_ids, .05, .1, filled=0)
    expected = select_hard_negative(real, real.detach(), curvature, shape_ids,
                                    curvature, shape_ids, .05, .1)
    assert all(torch.equal(a, b) for a, b in zip(cold, expected))


def training_smoke():
    """Exercise the actual training main on CPU with three synthetic batches per stage."""
    module = load_train()
    original_arange = torch.arange

    def arange(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_arange(*args, **kwargs)

    batches = [{"noisier_pat_pts": torch.randn(2, 500, 3),
                "stage1_pos_pat_pts": torch.randn(2, 500, 3),
                "fakeclean_pat_pts": torch.randn(2, 500, 3),
                "noisy_pat_ctr": torch.randn(2, 3), "shape_id": torch.tensor([0, 10001])}
               for _ in range(3)]
    with tempfile.TemporaryDirectory(prefix="geotriplet-smoke-") as temporary:
        output = Path(temporary) / "checkpoints"

        def run(stage, save, encoder=None):
            argv = ["train", "--stage", str(stage), "--epochs", "1",
                    "--save_dir", str(save), "--batch_size", "2", "--num_workers", "0"]
            if encoder is not None:
                argv += ["--encoder_ckpt", str(encoder)]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(module, "PairedPatchDataset", return_value=[]))
                stack.enter_context(patch.object(module, "DataLoader", return_value=batches))
                stack.enter_context(patch.object(torch.Tensor, "cuda", lambda t, *a, **kw: t))
                stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda m, *a, **kw: m))
                stack.enter_context(patch.object(torch, "arange", side_effect=arange))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                module.main()

        run(1, output)
        run(2, output, output / "stage1_latest.pth")
        for stage in (1, 2):
            checkpoint = torch.load(output / ("stage%d_latest.pth" % stage), map_location="cpu")
            assert checkpoint["epoch"] == 1
            assert checkpoint["config"]["bank_size"] == 1024
            assert checkpoint["config"]["negative_candidates"] == "filled_entries"
            assert all(torch.isfinite(t).all().item() for t in checkpoint["state_dict"].values())
        wrong = torch.load(output / "stage1_latest.pth", map_location="cpu")
        wrong["config"]["bank_size"] = 512
        invalid = Path(temporary) / "wrong_capacity.pth"
        torch.save(wrong, invalid)
        try:
            run(2, output.parent / "bad_handoff", invalid)
        except ValueError as error:
            assert "encoder_ckpt" in str(error)
        else:
            raise AssertionError("Stage 2 accepted a Stage 1 checkpoint with a different bank capacity")
        results = {"updates_per_stage": 3, "finite_weights": True,
                   "rejects_mismatched_handoff": True}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash_data", action="store_true")
    parser.add_argument("--training_smoke", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--report", default=str(OUTPUT_ROOT / "verification/report.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = {}
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    for entry in manifest["files"]:
        p = PROJECT_ROOT / entry["path"]
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == entry["bytes"], str(p)
        if args.hash_data:
            assert sha256(p) == entry["sha256"], str(p)
    checks["copied_files"] = len(manifest["files"])
    checks["data_sha256"] = "PASS" if args.hash_data else "not requested"
    index = json.loads((DATA_ROOT / "index.json").read_text())
    assert len(index["PUNet"]["test_cases"]) == 120
    for case in index["PUNet"]["test_cases"] + index["KinectFusion"]["cases"]:
        for key in ("input", "clean_pointcloud", "clean_mesh"):
            assert (DATA_ROOT / case[key]).is_file(), case[key]
    train_names = {p.stem for p in (DATA_ROOT / "PUNet/pointclouds/train/10000_poisson").glob("*.xyz")}
    assert len(train_names) == 40
    overlap = sorted(train_names.intersection(index["PUNet"]["test_shapes"]))
    assert overlap == index["PUNet"]["train_test_basename_overlap"]
    checks["train_test_basename_overlap"] = overlap
    for name in overlap:
        for resolution in index["PUNet"]["training_resolutions"]:
            train_file = DATA_ROOT / "PUNet/pointclouds/train" / resolution / (name + ".xyz")
            test_file = DATA_ROOT / "PUNet/pointclouds/test" / resolution / (name + ".xyz")
            assert sha256(train_file) != sha256(test_file)
    checks["same_basename_train_test_files_are_different"] = "PASS"
    order = json.loads((DATA_ROOT / "PUNet/file_order.json").read_text())["directories"]
    for folder, names in order.items():
        assert set(names) == {p.name for p in (DATA_ROOT / "PUNet" / folder).glob("*.xyz")}
    for names in index["PCPNet"]["splits"].values():
        assert all((DATA_ROOT / "PCPNet/pointclouds" / (n + ".xyz")).is_file() for n in names)
    assert len(index["KinectFusion"]["cases"]) == 6
    assert len(index["StreetScenes"]["cases"]) == 4
    assert not index["StreetScenes"]["ground_truth_available"]
    checks["splits_and_input_reference_pairs"] = "PASS"
    module = load_train()
    with tempfile.TemporaryDirectory(prefix="geotriplet-defaults-") as temporary:
        with patch.object(sys, "argv", ["train", "--stage", "1", "--save_dir", temporary]):
            config = module.parse_args()
    assert (config.bank_size, config.seed) == (1024, 2023)
    checks["training_defaults"] = "capacity 1024; filled entries only; score-denoise; seed 2023"
    model = GeoTripletDenoise().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        prediction, _ = model(torch.zeros(2, 500, 3))
    assert prediction.shape == (2, 3) and torch.isfinite(prediction).all()
    checks["strict_checkpoint_and_cpu_forward"] = {"state_dict_keys": len(checkpoint["state_dict"]),
                                                  "epoch": checkpoint.get("epoch"),
                                                  "sha256": sha256(args.checkpoint)}
    check_negative_candidates()
    checks["excludes_unwritten_entries_and_handles_cold_start"] = "PASS"
    if args.training_smoke:
        checks["training_smoke"] = training_smoke()
    report = {"status": "PASS", "device": "cpu", "torch": torch.__version__, "checks": checks}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
