# Dataset metadata

This delivery contains **no point clouds, meshes, normal references or dataset archives**. The following small metadata files are retained because they define the existing inference cases and preserve training order:

- `index.json`: relative input/reference paths and named splits for PUNet, PCPNet, KinectFusion and street scans.
- `PUNet/file_order.json`: the recorded shape order used by the PUNet loader. Keep this file with the actual PUNet root to preserve shape IDs and sample order when moving files between filesystems.
- `PUNet/examples/PUNet_shapes.txt`: the 20 PUNet test shape names.

These files contain names and paths, not point coordinates. The dataset payload and its file-integrity manifest are distributed separately.

## PUNet data layout

```text
PUNet/
├── file_order.json
├── pointclouds/train/{10000,30000,50000}_poisson/          # 40 clean shapes per resolution
├── pointclouds/test/{10000,50000}_poisson/                # 20 clean references per resolution
├── meshes/test/                                         # Clean reference meshes
└── examples/
    ├── PUNet_shapes.txt
    └── PUNet_{10000,50000}_poisson_{0.01,0.02,0.03}/      # Noisy test inputs
```

Training reads only the training point clouds and generates noise during loading. Final PUNet evaluation requires the noisy test inputs, clean test point clouds and reference meshes. Restore the supplied metadata alongside the actual data. Keep the existing file names and split assignments.

## Other datasets

`test_dataset.py` reads `index.json` and resolves its relative paths under this top-level `datasets/` directory:

- PCPNet inputs belong under `PCPNet/pointclouds/`; named splits are recorded in `index.json`.
- KinectFusion inputs belong under `KinectFusion/noisy/`, with reference paths recorded separately.
- Street inputs belong under `StreetScenes/RueMadame/`. No clean ground truth is supplied for these scans.

Once the corresponding data are supplied, use the included Stage 2 weights and `code/test_dataset.py --help` for inference options. The `--list_only` option inspects metadata without loading actual point clouds or a model.
