# GeoTripletDenoise

**Geometry-aware Triplet Learning for Unsupervised Point Cloud Denoising**

This package contains the reviewed training, inference and evaluation code, the two pretrained stage checkpoints, method documentation, environment specification and dataset metadata. Actual point clouds, meshes, normal references, training logs and generated results are not included. The Python source files and both checkpoint files are unchanged from the reviewed project.

## Layout

```text
GeoTripletDenoise/
├── README.md
├── code/
│   ├── train.py                # Two-stage training
│   ├── run_experiment.py       # Training, final inference and evaluation
│   ├── test_punet.py           # PUNet inference
│   ├── test_dataset.py         # PCPNet, KinectFusion and street-scan inference
│   ├── denoise.py              # Single-cloud inference
│   ├── evaluate.py             # PUNet CD, P2F and squared Hausdorff metrics
│   ├── network/                # Network architecture
│   ├── models/                 # Losses and negative selection
│   ├── datasets/               # Python data loader code
│   ├── utils/                  # Geometry, transforms, paths and compatibility
│   └── tools/                  # Full-project verification utility
├── datasets/                   # Metadata only; actual data are not included
│   ├── README.md
│   ├── index.json
│   └── PUNet/
│       ├── file_order.json
│       └── examples/PUNet_shapes.txt
├── checkpoints/
│   ├── stage1_latest.pth        # Stage 1 weights
│   └── stage2_latest.pth        # Stage 2 weights, used for inference
├── environment/
│   ├── README.md
│   ├── environment.yml
│   └── versions.json
├── docs/METHOD.md
└── SHA256SUMS                  # Package file checksums
```

`code/datasets/` contains Python modules required by training and inference. The separate top-level `datasets/` directory contains only small metadata files in this delivery. See [datasets/README.md](datasets/README.md) for the expected data layout. `checkpoints/` contains exactly the two pretrained files listed above.

## Environment and method

Use the **score-denoise** environment. The recorded runtime uses Python 3.8.20, PyTorch 1.9.0, CUDA 11.1 and PyTorch3D 0.5.0. See [environment/README.md](environment/README.md) for the environment files.

```bash
conda activate score-denoise
python code/train.py --help
python code/run_experiment.py --help
```

Training runs for **200 epochs per stage**, with seed **2023**, batch size **128** and four data workers. Stage 1 starts with **0 valid entries** in a bank of capacity **1024** and selects negatives only from filled entries, using real current-batch features at cold start. The bank fills progressively and then replaces older entries in a circular queue. Checkpoints are saved as **latest** after each epoch. There is no periodic validation or automatic best-checkpoint selection. See [docs/METHOD.md](docs/METHOD.md).

## Training and evaluation

After providing the PUNet data at the documented paths, run from the project root:

```bash
python code/run_experiment.py --gpu 0 --run_dir outputs/train
```

This trains both stages from scratch, then performs PUNet inference and evaluation with the run's final Stage 2 weights. Stage 2 initializes from the Stage 1 weights produced by that same run. Each run requires a fresh output directory. Its checkpoints, console logs, predictions and metrics are written under that output directory.

For data stored elsewhere, supply both roots:

```bash
python code/run_experiment.py --gpu 0 \
  --data_root /path/to/PUNet \
  --test_root /path/to/PUNet/examples/PUNet \
  --run_dir outputs/train
```

The external PUNet root must retain the supplied `file_order.json` and the documented split files to preserve the original loader order. `--test_root` is a prefix: the shape list is `<prefix>_shapes.txt`, and noisy inputs are under `<prefix>_<resolution>_<noise>/`.

## Inference with the included weights

The included weights are `checkpoints/stage1_latest.pth` and `checkpoints/stage2_latest.pth`. Inference uses the latter by default. After providing data, run PUNet inference and evaluation with:

```bash
python code/run_experiment.py --gpu 0 --skip_train \
  --run_dir outputs/evaluation
```

Single-cloud inference accepts explicit input, output and checkpoint paths:

```bash
CUDA_VISIBLE_DEVICES=0 python code/denoise.py \
  --input /path/to/noisy.xyz \
  --output outputs/denoised.xyz \
  --ckpt_path checkpoints/stage2_latest.pth
```

`code/tools/verify_project.py` checks the complete data-and-weight distribution. Its full checks require the separately supplied dataset payload and data manifest in addition to the included checkpoint. All command-line entry points support `--help` without the dataset payload.
