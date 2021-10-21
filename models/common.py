import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_points3d.core.common_modules import FastBatchNorm1d


def gather_neighbors(feature, neighbor_idx):
    """
    Gather neighbors according index.
    :param feature: [N, F]
    :type feature: Tensor
    :param neighbor_idx: [N/S, K]
    :type neighbor_idx: Tensor
    :return: [N/S, K, F]
    :rtype: Tensor
    """
    F, K = feature.shape[-1], neighbor_idx.shape[-1]
    neighbor_idx = neighbor_idx.reshape(-1, 1).expand(-1, F)
    neighbors = feature.gather(dim=0, index=neighbor_idx).reshape(-1, K, F)
    return neighbors.squeeze()


def nearest_interpolate(x, up_idx):
    up_idx = up_idx[:, 0]
    return gather_neighbors(x, up_idx)


def knn_interpolate(x, pos_x, pos_y, up_idx, k=3):
    """
    :param x: [S, F]
    :type x: Tensor
    :param pos_x: [S, D]
    :type pos_x: Tensor
    :param pos_y: [N, D]
    :type pos_y: Tensor
    :param up_idx: [N, K]
    :type up_idx: Tensor
    :return: [N, F]
    :rtype: Tensor
    """
    assert k <= up_idx.shape[-1]
    with torch.no_grad():
        up_idx = up_idx[:, :k]      # only use 3 nearest neighbors
        pos_x = gather_neighbors(pos_x, up_idx)                 # [N, 3, D]
        pos_y = pos_y.unsqueeze(1)
        dist = (pos_y - pos_x).pow(2).sum(dim=-1, keepdim=True).sqrt()
        weights = 1.0 / torch.clamp(dist, min=1e-16)            # [N, 3, 1]
        weights = weights / weights.sum(dim=1, keepdim=True)    # [N, 3 ,1]

    y = (gather_neighbors(x, up_idx) * weights).sum(dim=1)      # [N, F]

    return y


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, activation=None):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        bias = False if bn else True
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = FastBatchNorm1d(out_channels, momentum=0.02) if bn else None
        self.activation = activation

    def forward(self, x):
        x = self.lin(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __repr__(self):
        return '{}(in_features={}, out_features={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)


class Conv1x1(nn.Module):
    """
    Conv -> BN -> ReLU
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 padding_mode='zeros',
                 bn=True,
                 activation=None):
        super(Conv1x1, self).__init__()

        bias = False if bn else True
        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_fn(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            bias=bias,
                            padding_mode=padding_mode)

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.02) if bn else None
        if activation is not None:
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation = None

    def forward(self, x):
        """
        :param x: [B, C, N, K]
        :return: [B, C, N, K]
        """
        x = x.transpose(0, -1)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x.transpose(0, -1)

        return x


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, building_block):
        super(BottleneckResidualBlock, self).__init__()
        hidden_channels = out_channels // 4
        self.lin_in = MLP(in_channels, hidden_channels, activation=nn.ReLU())
        self.lin_out = MLP(hidden_channels, out_channels, activation=None)
        if in_channels != out_channels:
            self.shortcut = MLP(in_channels, out_channels, activation=None)
        else:
            self.shortcut = nn.Identity()

        self.building_block = building_block

    def forward(self, x, *args, **kwargs):
        residual = self.shortcut(x)
        x = self.lin_in(x)
        x = self.building_block(x, *args, **kwargs)
        x = self.lin_out(x)
        return F.leaky_relu(x + residual)


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.y = None
        self.y_hat = None
        self.loss = None

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, map_location=None):
        self.load_state_dict(torch.load(filename, map_location=map_location))

    def backward(self):
        self.loss.backward()
