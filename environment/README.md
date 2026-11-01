# score-denoise environment

This directory contains the working environment specification and imported version snapshot. The package includes pretrained weights, while actual datasets are supplied separately.

- `environment.yml`: a no-build Conda export with its machine-specific prefix removed.
- `versions.json`: the observed Python, PyTorch, CUDA and core package versions.

For an existing compatible environment, activate it and inspect an entry point:

```bash
conda activate score-denoise
python code/train.py --help
```

The recorded runtime uses Python 3.8.20, PyTorch 1.9.0, torchvision 0.10.0, CUDA toolkit 11.1, PyTorch3D 0.5.0, NumPy 1.22.4, SciPy 1.8.1, scikit-learn 1.3.2, pandas 2.0.3 and tqdm 4.67.1. Point Cloud Utils is recorded as 0.18.0 by Conda and 0.17.1 by its Python distribution metadata; both observations are preserved in the version records.

The project-local `code/utils/compat.py` fallback supplies the CPU single-center `ball_query` operation missing from this PyTorch3D version. It does not modify the installed package. The specification records the tested Linux environment; recreating it on another machine has not been validated in this packaging step.

The full `code/tools/verify_project.py` check is intended for the complete project after data and the data manifest have been provided; the checkpoint is already included in this package.
