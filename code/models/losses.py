import torch
import torch.nn.functional as F


def estimate_curvature_from_fake(p_fake):
    centroid = torch.mean(p_fake, dim=1, keepdim=True)
    centered = p_fake - centroid
    cov = torch.bmm(centered.transpose(1, 2), centered) / (p_fake.size(1) - 1)
    eigvals = torch.clamp(torch.linalg.eigvalsh(cov), min=0.0)
    return eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-6)


def fill_empty_rows(valid_mask, fallback_mask):
    empty_rows = valid_mask.sum(dim=1) == 0
    if empty_rows.any():
        valid_mask[empty_rows] = fallback_mask[empty_rows]
    return valid_mask


def select_hard_negative(
    f_a,
    bank_feats,
    bank_curvs,
    bank_ids,
    curv_a,
    shape_ids,
    curv_thresh,
    curv_fallback_thresh,
):
    diff_mask = bank_ids.unsqueeze(0) != shape_ids.unsqueeze(1)
    curv_diff = torch.abs(curv_a.unsqueeze(1) - bank_curvs.unsqueeze(0))
    valid_mask = (curv_diff < curv_thresh) & diff_mask
    valid_mask = fill_empty_rows(
        valid_mask, (curv_diff < curv_fallback_thresh) & diff_mask
    )
    valid_mask = fill_empty_rows(valid_mask, diff_mask)

    dist_mat = torch.cdist(f_a, bank_feats, p=2)
    dist_mat[~valid_mask] = float("inf")
    neg_idx = torch.argmin(dist_mat, dim=1)
    neg_curv_diff = curv_diff[
        torch.arange(curv_diff.size(0), device=curv_diff.device), neg_idx
    ]
    return bank_feats[neg_idx].clone().detach(), neg_curv_diff.detach()


def triplet_loss(f_a, f_p, f_n, margin=0.3, margin_scale=0.1):
    d_ap = F.pairwise_distance(f_a, f_p, p=2)
    d_an = F.pairwise_distance(f_a, f_n, p=2)
    dynamic_margin = margin + margin_scale * torch.exp(-d_an.detach())
    reliability_weight = torch.exp(-d_ap.detach())
    loss = (reliability_weight * F.relu(d_ap - d_an + dynamic_margin)).mean()
    stats = {
        "loss": loss.detach(),
        "d_ap": d_ap.detach().mean(),
        "d_an": d_an.detach().mean(),
        "margin": dynamic_margin.detach().mean(),
    }
    return loss, stats
