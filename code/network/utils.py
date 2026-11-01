import torch
import torch.nn as nn
from pytorch3d.ops import knn_points


def knn_group(x: torch.FloatTensor, idx: torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    _, knn_idx, _ = knn_points(pos, query, K=k + offset, return_nn=False)
    return knn_idx[:, :, offset:]


class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bn=0, activation=None):
        super().__init__()
        assert with_bn in [0, 1, 2]
        self.with_bn = with_bn > 0 and activation is not None

        self.linear = nn.Linear(in_features, out_features)

        if self.with_bn:
            if with_bn == 2:
                self.bn = nn.BatchNorm2d(out_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        """
        x: (*, C)
        y: (*, C)
        """
        y = self.linear(x)
        if self.with_bn:
            if x.dim() == 2:  # (B, C)
                y = self.activation(self.bn(y))
            elif x.dim() == 3:  # (B, N, C)
                y = self.activation(self.bn(y.transpose(1, 2))).transpose(1, 2)
            elif x.dim() == 4:  # (B, H, W, C)
                y = self.activation(self.bn(y.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        else:
            y = self.activation(y)
        return y
