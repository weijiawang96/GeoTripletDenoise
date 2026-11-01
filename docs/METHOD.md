# Experiment settings

Paper: **Geometry-aware Triplet Learning for Unsupervised Point Cloud Denoising**.

The network combines two feature encoders, geometry-aware triplet learning and denoising regression. Training uses a memory bank with capacity 1024 and selects negatives only from filled entries, with a real-batch fallback when required.

| Setting | Value |
| --- | --- |
| Environment | score-denoise |
| Stages | 200 epochs of triplet pretraining, then 200 epochs of denoising |
| Bank capacity | 1024 |
| Negative candidates | Filled entries only; real current-batch fallback at cold start |
| Training seed | 2023 |
| Batch size / workers | 128 / 4 |
| Patch points / radius | 500 / 0.05 of bounding-box diagonal |
| Encoder neighbors | 16 |
| Optimizer | Adam, learning rate 0.0001 |
| Schedule | Milestones 80, 120, 160; factor 0.1, independently per stage |
| Stage 1 loss | `(left_triplet + 0.5 * right_triplet) / 1.5` |
| Positive | Once-noisy patch, in the anchor's PCA frame |
| Negative selection | Curvature-aware hard negative, excluding the same shape |
| Margin / margin scale | 0.3 / 0.1 |
| Stage 2 loss | Original MSE term plus 0.0005 times the original repulsion term |
| Stage 2 trainable parameters | Joint finetuning except the two projection heads |
| Checkpoint policy | Save latest each epoch; no validation or best-score selection |

Stage 1 starts with **0 valid entries**. Negative selection happens before the current batch is written to the bank. At cold start, it uses real detached features from the current batch, excluding the same shape. Each update then writes the batch's left/right features, curvature values and shape IDs and increases the valid entry count up to 1024.

With the default batch size of 128, the valid entry count reaches 1024 after eight updates; the ninth update starts with 1024 historical entries. The count and write pointer carry across epochs. Once capacity is reached, the bank replaces older entries in a circular queue. If the existing history has no different-shape candidate for any anchor, real detached current-batch features are appended to the candidate set. Unwritten entries never participate in negative selection.

The training bank is not part of the inference network. Denoising loads `checkpoints/stage2_latest.pth` by default.

The score-denoise PyTorch3D version lacks `ball_query`. A project-local CPU single-center fallback supplies the training patch query; the dispatcher uses the native operator if available. Network neighbor operations continue to use PyTorch3D. No installed package is modified.

## PUNet inference and metrics

Inference uses 10k and 50k point clouds, 20 shapes each, at noise levels 0.01, 0.02 and 0.03. Pass counts are 1, 1 and 2 respectively. Every pass writes and reloads float32 XYZ. Predictions are transformed from the patch PCA frame back to world coordinates. Inference uses seed 2023, set after imports.

Metrics use the original code: unit-sphere CD and P2F multiplied by 10,000, and squared Hausdorff distance multiplied by 1,000. Comparisons should use the recorded score-denoise environment because PyTorch3D versions can affect P2F. Inference-only commands for PCPNet, KinectFusion and street scans are supplied separately; they do not silently apply the PUNet metric protocol.
