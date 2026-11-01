"""Select negative candidates from written history with real-batch fallbacks."""
import torch
from models.losses import select_hard_negative


def select_negative(f_a, bank_feats, bank_curvs, bank_ids, curv_a, shape_ids,
                    curv_thresh, curv_fallback_thresh, *, filled):
    if filled:
        feats = bank_feats[:filled]
        curvs = bank_curvs[:filled]
        ids = bank_ids[:filled]
        # If history contains no different shape for a row, add real features
        # from this batch as a fallback. Unwritten bank entries never participate.
        if not (ids.unsqueeze(0) != shape_ids.unsqueeze(1)).any(dim=1).all():
            feats = torch.cat((feats, f_a.detach()))
            curvs = torch.cat((curvs, curv_a.detach()))
            ids = torch.cat((ids, shape_ids.detach()))
    else:
        feats, curvs, ids = f_a.detach(), curv_a.detach(), shape_ids.detach()
    if not (ids.unsqueeze(0) != shape_ids.unsqueeze(1)).any(dim=1).all():
        raise RuntimeError('No real negative with a different shape ID in history or batch')
    return select_hard_negative(f_a, feats, curvs, ids, curv_a, shape_ids,
                                curv_thresh, curv_fallback_thresh)
