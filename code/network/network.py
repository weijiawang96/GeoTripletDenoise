"""Geometry-aware triplet denoising network; checkpoint parameter names are preserved."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import LinearLayer, get_knn_idx, knn_group

BN1d = 1
BN2d = 2


def _ensure_bn3(pos):
    return pos if pos.size(-1) == 3 else pos.transpose(1, 2)


class GraphConvPure(nn.Module):
    def __init__(self, in_channels, num_fc_layers, growth_rate, knn=16):
        super().__init__()
        self.knn = knn
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.layer_first = LinearLayer(
            in_channels * 3, growth_rate, with_bn=BN2d, activation="relu"
        )
        self.layers_mid = nn.ModuleList(
            [
                LinearLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    with_bn=BN2d,
                    activation="relu",
                )
                for i in range(1, num_fc_layers - 1)
            ]
        )
        self.layer_last = LinearLayer(
            in_channels + (num_fc_layers - 1) * growth_rate,
            growth_rate,
            with_bn=False,
            activation=None,
        )

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def forward(self, x, pos, knn_idx=None):
        if knn_idx is None:
            knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        knn_feat = knn_group(x, knn_idx)
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)

        y = torch.cat(
            [
                self.layer_first(edge_feat),
                x.unsqueeze(-2).repeat(1, 1, self.knn, 1),
            ],
            dim=-1,
        )
        for layer in self.layers_mid:
            y = torch.cat([layer(y), y], dim=-1)
        y = torch.cat([self.layer_last(y), y], dim=-1)
        y, _ = y.max(dim=-2, keepdim=False)
        return y, knn_idx


class EncodeNetBrain(nn.Module):
    """Right branch with the RightDeep global projection: 60 -> 256 -> 1024."""

    def __init__(self, knn=16, local_dim=64, global_dim=1024):
        super().__init__()
        self.trans = nn.ModuleList(
            [
                LinearLayer(3 if i == 0 else 60, 24, with_bn=BN1d, activation="relu")
                for i in range(4)
            ]
        )
        self.convs = nn.ModuleList(
            [GraphConvPure(24, 3, 12, knn=knn) for _ in range(4)]
        )
        enc_dim = self.convs[-1].out_channels
        self.local_proj = nn.Sequential(
            nn.Conv1d(enc_dim, local_dim, 1),
            nn.BatchNorm1d(local_dim),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Conv1d(enc_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, global_dim, 1),
            nn.BatchNorm1d(global_dim),
            nn.ReLU(),
        )

    def forward(self, pos_b3n):
        pos_bn3 = _ensure_bn3(pos_b3n)
        x, pos = pos_bn3, pos_bn3
        knn_idx = None
        for i in range(4):
            x = self.trans[i](x)
            x, knn_idx = self.convs[i](x, pos=pos, knn_idx=knn_idx)
        feat = x.transpose(2, 1)
        f_local = self.local_proj(feat)
        f_global = torch.max(self.global_proj(feat), 2, keepdim=True)[0]
        return f_local, f_global


class BasePointNetLeftBrain(nn.Module):
    """Left branch with the LeftProj token projection: 1024 -> 64."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.local_proj = nn.Sequential(
            nn.Conv1d(1024, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        f_high = self.net(x)
        f_local = self.local_proj(f_high)
        f_global = torch.max(f_high, 2, keepdim=True)[0]
        return f_local, f_global


class CrossAttentionFusionV2(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_right_local, f_left_local):
        q_in = f_right_local.transpose(1, 2)
        kv_in = f_left_local.transpose(1, 2)
        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)
        attn_out, _ = self.mha(q, k, v)
        return self.norm(q_in + attn_out).transpose(1, 2)


def make_proj_head():
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
    )


class GeoTripletDenoise(nn.Module):
    """LeftProj + RightDeep network with triplet heads for both encoders."""

    def __init__(self, encode_knn=16):
        super().__init__()
        self.encoder_trip = BasePointNetLeftBrain()
        self.encoder_reg = EncodeNetBrain(knn=encode_knn)
        self.left_proj_head = make_proj_head()
        self.right_proj_head = make_proj_head()

        self.conv_g = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.attn = CrossAttentionFusionV2(embed_dim=64)
        self.conv_f = nn.Sequential(
            nn.Conv1d(323, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def _to_b3n(self, pos_bn3):
        return pos_bn3.transpose(2, 1) if pos_bn3.size(-1) == 3 else pos_bn3

    def left_triplet_feat(self, pos_bn3):
        pos_b3n = self._to_b3n(pos_bn3)
        _, f_gt = self.encoder_trip(pos_b3n)
        feat = self.left_proj_head(f_gt.view(f_gt.size(0), -1))
        return F.normalize(feat, p=2, dim=1)

    def right_triplet_feat(self, pos_bn3):
        pos_b3n = self._to_b3n(pos_bn3)
        _, f_gr = self.encoder_reg(pos_b3n)
        feat = self.right_proj_head(f_gr.view(f_gr.size(0), -1))
        return F.normalize(feat, p=2, dim=1)

    def forward(self, pos_bn3, return_feat=False):
        pos_b3n = self._to_b3n(pos_bn3)
        f_lt, f_gt = self.encoder_trip(pos_b3n)

        if return_feat:
            feat = self.left_proj_head(f_gt.view(f_gt.size(0), -1))
            return None, F.normalize(feat, p=2, dim=1)

        f_lr, f_gr = self.encoder_reg(pos_b3n)
        bsz, _, num_pts = pos_b3n.shape
        g = self.conv_g(torch.cat([f_gt, f_gr], dim=1)).expand(-1, -1, num_pts)
        f_guided = self.attn(f_lr, f_lt)
        x = self.conv_f(torch.cat([pos_b3n, f_guided, g], dim=1))

        ctr_idx = torch.argmin((pos_b3n**2).sum(dim=1), dim=1)
        ctr_feat = x[torch.arange(bsz, device=pos_b3n.device), :, ctr_idx]
        patch_feat = torch.max(x, dim=2)[0]
        return self.mlp_out(torch.cat([patch_feat, ctr_feat], dim=1)), None
