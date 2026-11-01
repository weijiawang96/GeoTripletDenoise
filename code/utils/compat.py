"""Local compatibility for CPU single-center patch queries in older PyTorch3D."""
import torch
import pytorch3d.ops


def ball_query_cpu_fallback(p1, p2, K, radius):
    if p1.device.type != 'cpu' or p2.device.type != 'cpu' or p1.shape[:2] != (1, 1):
        raise ValueError('The compatibility query supports the training CPU single-center path only')
    delta = p2[0] - p1[0, 0]
    # Accumulate the three coordinates in the same order as the native CPU op.
    d2 = delta[:, 0] * delta[:, 0]
    d2 = d2 + delta[:, 1] * delta[:, 1]
    d2 = d2 + delta[:, 2] * delta[:, 2]
    # The C++ op receives radius as a float before multiplying it by itself.
    r = torch.tensor(float(radius), dtype=p1.dtype)
    chosen = torch.nonzero(d2 < r * r, as_tuple=False).flatten()[:K]
    idx = torch.full((1, 1, K), -1, dtype=torch.long)
    points = torch.zeros((1, 1, K, 3), dtype=p2.dtype)
    distances = torch.zeros((1, 1, K), dtype=p2.dtype)
    count = chosen.numel()
    idx[0, 0, :count] = chosen
    points[0, 0, :count] = p2[0, chosen]
    distances[0, 0, :count] = d2[chosen]
    return distances, idx, points


def ball_query(p1, p2, K, radius):
    native = getattr(pytorch3d.ops, 'ball_query', None)
    if native is not None:
        return native(p1, p2, K=K, radius=radius)
    return ball_query_cpu_fallback(p1, p2, K=K, radius=radius)
