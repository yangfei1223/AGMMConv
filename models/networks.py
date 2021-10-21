# -*- coding:utf-8 -*-
'''
@Time : 2020/9/13 下午4:09
@Author: yangfei
@File : networks.py
'''
from torch_geometric.nn import global_mean_pool
from .gmm_conv import *

class ResidualGMMNet(Base):
    def __init__(self,
                 in_channels,
                 n_classes,
                 num_kernels,
                 layers,
                 class_weights=None,
                 ignore_index=-1,
                 is_pooling=True):
        super(ResidualGMMNet, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.is_pooling = is_pooling
        self.internal_loss = None
        self.external_loss = None

        self.encoder = nn.ModuleList()
        for i in range(len(layers)):
            in_dim = in_channels if i == 0 else layers[i - 1]
            out_dim = layers[i]
            block = nn.ModuleList([
                ResGMMConvBBlock(in_dim, out_dim, num_kernels, activation=nn.LeakyReLU(0.1)),
                ResGMMConvBBlock(out_dim, out_dim, num_kernels, activation=nn.LeakyReLU(0.1))
            ])
            self.encoder.add_module('Conv Block {}'.format(i), block)

        self.classifier = nn.Sequential(
            MLP(layers[-1], layers[-1] * 2, activation=nn.LeakyReLU(0.1)),
            nn.Dropout(p=0.5),
            nn.Linear(layers[-1] * 2, n_classes)
        )

    def forward(self, data):
        x, ms = data.x, data.multiscale

        # encoder
        if self.is_pooling:
            for i, block in enumerate(self.encoder):
                if i > 0:   # pooling
                    x = gather_neighbors(x, ms[i - 1].sub_idx)
                    x = x.max(dim=1)[0]
                x = block[0](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
                x = block[1](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
        else:
            for i, block in enumerate(self.encoder):
                x = block[0](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx) if i == 0 \
                    else block[0](x, ms[i - 1].pos, ms[i].pos, ms[i - 1].sub_idx)
                x = block[1](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)

        x = global_mean_pool(x, ms[-1].batch)

        self.y_hat = self.classifier(x)
        self.y = data.y
        return self.y_hat

    def _get_internal_loss(self):
        total_loss = torch.tensor([0], device=self.y_hat.device, dtype=torch.float32)
        for block in self.encoder:
            for layer in block:
                total_loss += layer.gmm_conv.get_internal_loss()
        return total_loss

    def compute_loss(self, lambda_internal=1e-2):
        self.internal_loss = self._get_internal_loss()
        class_weights = self.class_weights.to(self.y.device) if self.class_weights is not None else None
        self.external_loss = F.cross_entropy(self.y_hat, self.y,  weight=class_weights, ignore_index=self.ignore_index)
        self.loss = self.external_loss + lambda_internal * self.internal_loss
        return self.loss

class ResidualGMMSegNet(Base):
    def __init__(self,
                 in_channels,
                 n_classes,
                 num_kernels,
                 layers,
                 class_weights=None,
                 ignore_index=-1,
                 is_transpose=False):
        super(ResidualGMMSegNet, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.is_transpose = is_transpose
        self.internal_loss = None
        self.external_loss = None

        self.encoder = nn.ModuleList()
        for i in range(len(layers)):
            in_dims = in_channels if i == 0 else layers[i - 1]
            out_dims = layers[i]
            block = nn.ModuleList([
                ResGMMConvBBlock(in_dims, out_dims, num_kernels),
                ResGMMConvBBlock(out_dims, out_dims, num_kernels),
            ])
            self.encoder.add_module('Conv Block {}'.format(i), block)

        self.decoder = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_dims = layers[- i - 1]
            out_dims = layers[- i - 2]
            block = GMMConvTranspose(in_dims, out_dims, num_kernels, activation=nn.LeakyReLU(0.1)) \
                if is_transpose else Upsample(in_dims, out_dims, activation=nn.LeakyReLU(0.1))
            self.decoder.add_module('Upsample Block{}'.format(i), block)

        self.classifier = nn.Sequential(
            MLP(layers[0], layers[0] * 4, activation=nn.LeakyReLU(0.1)),
            nn.Linear(layers[0] * 4, n_classes)
        )

    def forward(self, data):
        x, ms = data.x, data.multiscale

        # encoder
        encoder_stack = []
        for i, block in enumerate(self.encoder):
            if i == 0:
                x = block[0](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
            else:
                x = block[0](x, ms[i - 1].pos, ms[i].pos, ms[i - 1].sub_idx)
            x = block[1](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
            encoder_stack.append(x)

        # decoder
        for j, block in enumerate(self.decoder):
            x = block(x, ms[-j - 1].pos, ms[-j - 2].pos, ms[-j - 2].up_idx)
            x = x + encoder_stack[-j - 2]

        self.y_hat = self.classifier(x)
        self.y = data.y
        return self.y_hat

    def _get_internal_loss(self):
        total_loss = torch.tensor([0], device=self.y_hat.device, dtype=torch.float32)
        for block in self.encoder:
            for layer in block:
                total_loss += layer.gmm_conv.get_internal_loss()
        if self.is_transpose:
            for layer in self.decoder:
                total_loss += layer.gmm_conv.get_internal_loss()
        return total_loss

    def compute_loss(self, lambda_internal=0.01):
        self.internal_loss = self._get_internal_loss()
        class_weights = self.class_weights.to(self.y.device) if self.class_weights is not None else None
        self.external_loss = F.cross_entropy(self.y_hat, self.y,  weight=class_weights, ignore_index=self.ignore_index)
        self.loss = self.external_loss + lambda_internal * self.internal_loss
        return self.loss

class ResudualGMMNormalEstimation(Base):
    def __init__(self,
                 in_channels,
                 num_kernels,
                 layers,
                 is_transpose=True):
        super(ResudualGMMNormalEstimation, self).__init__()
        self.is_transpose = is_transpose
        self.internal_loss = None
        self.external_loss = None

        self.encoder = nn.ModuleList()
        for i in range(len(layers)):
            in_dims = in_channels if i == 0 else layers[i - 1]
            out_dims = layers[i]
            block = nn.ModuleList([
                ResGMMConvBBlock(in_dims, out_dims, num_kernels),
                ResGMMConvBBlock(out_dims, out_dims, num_kernels),
            ])
            self.encoder.add_module('Conv Block {}'.format(i), block)

        self.decoder = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_dims = layers[- i - 1]
            out_dims = layers[- i - 2]
            block = GMMConvTranspose(in_dims, out_dims, num_kernels, activation=nn.LeakyReLU(0.1)) \
                if is_transpose else Upsample(in_dims, out_dims, activation=nn.LeakyReLU(0.1))
            self.decoder.add_module('Upsample Block{}'.format(i), block)

        self.classifier = nn.Sequential(
            MLP(layers[0], layers[0] * 4, activation=nn.LeakyReLU(0.1)),
            nn.Dropout(p=0.5),
            nn.Linear(layers[0] * 4, 3)
        )

    def forward(self, data):
        x, batch, ms = data.x, data.batch, data.multiscale

        # encoder
        encoder_stack = []
        for i, block in enumerate(self.encoder):
            if i == 0:
                x = block[0](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
            else:
                x = block[0](x, ms[i - 1].pos, ms[i].pos, ms[i - 1].sub_idx)
            x = block[1](x, ms[i].pos, ms[i].pos, ms[i].neighbor_idx)
            encoder_stack.append(x)

        # decoder
        for j, block in enumerate(self.decoder):
            x = block(x, ms[-j - 1].pos, ms[-j - 2].pos, ms[-j - 2].up_idx)
            x = x + encoder_stack[-j - 2]

        y_hat = self.classifier(x)
        self.y_hat = F.normalize(y_hat, p=2, dim=-1)
        self.y = data.norm
        return self.y_hat

    def _get_internal_loss(self):
        total_loss = torch.tensor([0], device=self.y_hat.device, dtype=torch.float32)
        for block in self.encoder:
            for layer in block:
                total_loss += layer.gmm_conv.get_internal_loss()
        if self.is_transpose:
            for layer in self.decoder:
                total_loss += layer.gmm_conv.get_internal_loss()
        return total_loss

    def compute_loss(self, lambda_internal=1e-2):
        self.internal_loss = self._get_internal_loss()
        self.external_loss = (1. - F.cosine_similarity(self.y_hat, self.y, dim=-1).abs()).mean()
        self.loss = self.external_loss + lambda_internal * self.internal_loss
        return self.loss






