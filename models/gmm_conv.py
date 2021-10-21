import math
from .common import *
EPS = 1e-5

class MixtureDensityNetwork(nn.Module):
    def __init__(self, hidden_channels, num_kernels):
        super(MixtureDensityNetwork, self).__init__()
        self.num_kernels = num_kernels
        self.local_nn = MLP(3, hidden_channels, activation=nn.LeakyReLU(0.1))
        self.global_nn = nn.Sequential(
            MLP(hidden_channels * 2, hidden_channels * 2, activation=nn.LeakyReLU(0.1)),
            MLP(hidden_channels * 2, num_kernels * 7, bn=False, activation=None)
        )

        self.internal_loss = None

    @staticmethod
    def _regularization_loss(a, thresh):
        norm = a.norm(dim=-1)
        choice = norm > thresh
        return norm[choice].mean() if choice.any() else 0

    @staticmethod
    def _likelihood_loss(pi, sigma2, w):
        z = (2 * math.pi * sigma2).sqrt().prod(dim=-1)          # [N, H, 1]
        w = pi * (w / z)
        return -torch.log(w.sum(dim=1) + 1e-10).sum(dim=-1).mean()

    def forward(self, pos_in, pos_out, neighbor_idx):
        N, K, H = neighbor_idx.shape[0], neighbor_idx.shape[1], self.num_kernels
        neighbors = gather_neighbors(pos_in, neighbor_idx)
        neighbors = neighbors - pos_out.unsqueeze(1)                                # [N, K, D]

        x = self.local_nn(neighbors)                                                # [N, K, F]
        x_max = x.mean(dim=1)                                                       # [N, F]
        x_mean = x.max(dim=1)[0]                                                    # [N, F]
        x = torch.cat([x_max, x_mean], dim=-1)
        x = self.global_nn(x)                                                       # [N, H * 6]
        pi, mu, sigma = x.reshape(N, H, -1).split([1, 3, 3], dim=-1)                # [N, H, 1], [N, H, 3], [N, H, 3]
        pi = pi.softmax(dim=1)                                                      # [N, H, 1]

        dist2 = (neighbors.reshape(N, 1, K, -1) - mu.reshape(N, H, 1, -1)).pow(2)   # [N, H, K, D]
        sigma2 = sigma.reshape(N, H, 1, -1).pow(2) + EPS                            # [N, H, 1, D]
        dist2 = (dist2 / sigma2).sum(dim=-1)                                        # [N, H, K]
        w = torch.exp(-dist2 / 2)

        # regularization loss
        thresh = neighbors.norm(dim=-1, keepdim=True).max(dim=1)[0]
        loss1 = self._regularization_loss(mu, thresh=thresh)
        loss2 = self._regularization_loss(sigma, thresh=thresh / 2.)
        # likelihood loss
        loss3 = self._likelihood_loss(pi, sigma2, w)
        self.internal_loss = loss1 + loss2 + 1e-2 * loss3

        return pi, w


class AGMMConv(nn.Module):
    """
    GMM Convolution for point cloud, can be stride or non-stride.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 add_ones=True,
                 bn=True,
                 activation=None,
                 use_weights=True):
        super(AGMMConv, self).__init__()
        self.add_ones = add_ones
        self.use_weights = use_weights

        self.generator = MixtureDensityNetwork(hidden_channels=32, num_kernels=num_kernels)
        if self.use_weights:
            self.weights = nn.Parameter(torch.zeros((num_kernels, in_channels + add_ones * 1, out_channels),
                                                    dtype=torch.float32), requires_grad=True)
            self.bn = FastBatchNorm1d(out_channels, momentum=0.02) if bn else None
        else:
            self.bn = FastBatchNorm1d(out_channels + add_ones * 1, momentum=0.02) if bn else None
        self.activation = activation

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_weights:
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def get_internal_loss(self):
        return self.generator.internal_loss

    def forward(self, x, pos_in, pos_out, neighbor_idx):
        """
        :param x: [N_in, F_in]
        :type x: Tensor
        :param pos_in: [N_in, D]
        :type pos_in: Tensor
        :param pos_out: [N_out, D]
        :type pos_out: Tensor
        :param neighbor_idx: [N_out, K]
        :type neighbor_idx: Tensor
        :return: [N_out, F]
        :rtype: Tensor
        """
        pi, w = self.generator(pos_in, pos_out, neighbor_idx)       # [N, H, K]

        if self.add_ones:
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            x = torch.cat([x, ones], dim=-1)
        x = gather_neighbors(x, neighbor_idx)                       # [N, K, F]

        x = w.matmul(x).transpose(0, 1)                             # [H, N, F_in], kernel association
        if self.use_weights:
            x = x.matmul(self.weights)                              # [H, N, F_out], convolution
        x = (pi.transpose(0, 1) * x).sum(dim=0)                     # [N, F_out], kernel selection

        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class ResGMMConvBBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 add_ones=True,
                 bn=True,
                 activation=None,
                 use_weights=True):
        super(ResGMMConvBBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        hidden_channels = out_channels // 4
        if in_channels != out_channels:
            self.shortcut = MLP(in_channels, out_channels, bn=bn, activation=None)
        else:
            self.shortcut = nn.Identity()

        self.lin_in = MLP(in_channels, hidden_channels, bn=bn, activation=nn.LeakyReLU(0.1))
        if use_weights:
            self.lin_out = MLP(hidden_channels, out_channels, bn=bn, activation=None)
        else:
            self.lin_out = MLP(hidden_channels + add_ones * 1, out_channels, bn=bn, activation=None)

        self.gmm_conv = AGMMConv(in_channels=hidden_channels, out_channels=hidden_channels, num_kernels=num_kernels,
                                add_ones=add_ones, bn=bn, activation=activation, use_weights=use_weights)

    def forward(self, x, pos_in, pos_out, neighbor_idx):
        if pos_in.shape[0] != pos_out.shape[0]:
            residual = gather_neighbors(x, neighbor_idx)
            residual = residual.max(dim=1)[0]
        else:
            residual = x
        residual = self.shortcut(residual)

        x = self.lin_in(x)
        x = self.gmm_conv(x, pos_in, pos_out, neighbor_idx)
        x = self.lin_out(x)

        return F.leaky_relu(x + residual, negative_slope=0.1)

    def __repr__(self):
        return '{}(in_features={}, out_features={}, num_kernels={})'.format(self.__class__.__name__,
                                                                            self.in_channels,
                                                                            self.out_channels,
                                                                            self.num_kernels)


class GMMConvTranspose(nn.Module):
    """
    Transpose GMMConv with Bottleneck for efficient computing.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 add_ones=False,
                 has_bottleneck=True,
                 bn=True,
                 activation=None):
        super(GMMConvTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.has_bottleneck = has_bottleneck
        if has_bottleneck:
            hidden_channels = out_channels // 4
            self.lin_in = MLP(in_channels, hidden_channels, bn=bn, activation=activation)
            self.gmm_conv = AGMMConv(in_channels=hidden_channels, out_channels=hidden_channels, num_kernels=num_kernels,
                                    add_ones=add_ones, bn=bn, activation=activation)
            self.lin_out = MLP(hidden_channels, out_channels, bn=bn, activation=activation)
        else:
            self.gmm_conv = AGMMConv(in_channels=in_channels, out_channels=out_channels, num_kernels=num_kernels,
                                    add_ones=add_ones, bn=bn, activation=activation)

    def forward(self, x, pos_in, pos_out, neighbor_idx):
        if self.has_bottleneck:
            x = self.lin_in(x)
            x = self.gmm_conv(x, pos_in, pos_out, neighbor_idx)
            x = self.lin_out(x)
        else:
            x = self.gmm_conv(x, pos_in, pos_out, neighbor_idx)
        return x

    def __repr__(self):
        return '{}(in_features={}, out_features={}, num_kernels={})'.format(self.__class__.__name__,
                                                                            self.in_channels,
                                                                            self.out_channels,
                                                                            self.num_kernels)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(Upsample, self).__init__()

        self.mlp = MLP(in_channels, out_channels, activation=activation)

    def forward(self, x, pos_in, pos_out, up_idx):
        x = knn_interpolate(x, pos_in, pos_out, up_idx)
        x = self.mlp(x)
        return x

















