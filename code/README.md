# Code

The code is organized into training, inference, network, loss, data loader and utility modules.

| Path | Purpose |
| --- | --- |
| `train.py` | Two-stage training, bank capacity 1024, filled-only negatives, latest checkpoint each epoch |
| `run_experiment.py` | Stage 1, Stage 2, PUNet inference and evaluation |
| `test_punet.py` | Original PUNet inference protocol |
| `test_dataset.py` | Manifest-based inference on PCPNet, KinectFusion and street scans |
| `denoise.py` | Inference on one XYZ file |
| `evaluate.py` | Original PUNet CD, P2F and squared Hausdorff metrics |
| `network/` | PointNet and graph encoders, projection heads, attention and regression |
| `models/` | Triplet loss, curvature, negative selection and metric operations |
| `datasets/` | Python dataset loaders; actual data are supplied separately under `../datasets/` |
| `utils/` | Transforms, geometry, progress, project paths, checkpoint checks and PyTorch3D compatibility |
| `tools/verify_project.py` | CPU verification, checkpoint loading and optional full data hashing |

See the project-level README for commands. The included weights are `checkpoints/stage1_latest.pth` and `checkpoints/stage2_latest.pth`. Actual dataset files are supplied separately.
