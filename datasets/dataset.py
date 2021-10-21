# -*- coding:utf-8 -*-
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader

from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.transforms import Compose, GridSampling, NormalizeScale
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import fps

from torch_points_kernels.torchpoints import ball_query, furthest_point_sample
from torch_points_kernels.torchpoints import tpcpu, tpcuda
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.multiscale_data import MultiScaleData, MultiScaleBatch
from torch_points3d.core.data_transform import GridSampling3D

from utils import cpp_subsampling, nearest_neighbors


def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    elif features is None:
        return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)


def knn_search(support_points, query_points, k):
    neighbor_idx = nearest_neighbors.knn(support_points, query_points, k, omp=True)
    return torch.from_numpy(neighbor_idx)


def radius_search(support_points, query_points, radius, max_num):
    neighbor_idx = nearest_neighbors.radius(support_points, query_points, radius, max_num, omp=True)
    return torch.from_numpy(neighbor_idx)


def batch_knn_search(support_points, query_points, k, support_batch=None, query_batch=None):
    if support_points.dim() == 2:
        neighbor_idx = nearest_neighbors.knn_sparse_batch(support_points, query_points, support_batch, query_batch, k, omp=True)
    elif support_points.dim() == 3:
        neighbor_idx = nearest_neighbors.knn_dense_batch(support_points, query_points, k, omp=True)
    else:
        raise ValueError('Only [N, 3] or [B, N, 3] is supported!')
    return torch.from_numpy(neighbor_idx)


def batch_radius_search(support_points, query_points, radius, max_num, support_batch=None, query_batch=None):
    if support_points.dim() == 2:
        neighbor_idx = nearest_neighbors.radius_sparse_batch(support_points, query_points, support_batch, query_batch, radius, max_num, omp=True)
    elif support_points.dim() == 3:
        neighbor_idx = nearest_neighbors.radius_dense_batch(support_points, query_points, radius, max_num, omp=True)
    else:
        raise ValueError('Only [N, 3] or [B, N, 3] is supported!')
    return torch.from_numpy(neighbor_idx)


def grid_sampling_sparse(pos, grid_size, batch=None):
    coords = torch.round(pos / grid_size)
    if batch is None:
        cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
    else:
        cluster = voxel_grid(coords, batch, 1)
    cluster, unique_pos_indices = consecutive_cluster(cluster)
    return cluster, unique_pos_indices


class MultiScaleTransform(object):
    def __init__(self,
                 kernel_size,
                 grid_size,
                 radius,
                 sample_ratio,
                 search_method='knn',
                 sample_method='grid',
                 ):
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.radius = radius
        self.sample_ratio = sample_ratio
        self.num_scales = len(kernel_size)
        assert search_method in ['knn', 'radius']
        assert sample_method in ['random', 'grid', 'fps']
        self.search_method = search_method
        self.sample_method = sample_method

    def __call__(self, data: Data) -> MultiScaleData:
        data.contiguous()
        multiscale = []
        pos = data.pos  # [N, 3]
        for i in range(self.num_scales):
            if self.search_method.lower() == 'knn':
                neighbor_idx = knn_search(pos, pos, self.kernel_size[i])
            elif self.search_method.lower() == 'radius':
                neighbor_idx = radius_search(pos, pos, self.radius[i], self.kernel_size[i])
            else:
                raise NotImplementedError('Only `knn` and `radius` are implemented!')
            if self.sample_method.lower() == 'random':
                sample_num = int(pos.shape[0] * self.sample_ratio[i])
                choice = torch.randperm(pos.shape[1])[:sample_num]
                sub_pos = pos[:, choice, :]
                sub_idx = neighbor_idx[:, choice, :]
            elif self.sample_method.lower() == 'fps':
                choice = fps(pos.cuda(), ratio=self.sample_ratio[i]).cpu()
                sub_pos = pos[choice]
                sub_idx = neighbor_idx[choice]
            elif self.sample_method.lower() == 'grid':
                cluster, indices = grid_sampling_sparse(pos, self.grid_size[i])
                sub_pos = scatter_mean(pos, cluster, dim=0)
                sub_idx = neighbor_idx[indices]
            else:
                raise NotImplementedError('Only `random`, `fps`, and `grid` sampling method are implemented!')
            up_idx = knn_search(sub_pos, pos, 1)  # [N, 1], find the nearest neighbor
            multiscale.append(Data(pos=pos, neighbor_idx=neighbor_idx, sub_idx=sub_idx, up_idx=up_idx))
            pos = sub_pos

        ms_data = MultiScaleData.from_data(data)
        ms_data.multiscale = multiscale
        return ms_data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class BaseDataset:
    def __init__(self,
                 kernel_size,
                 dilation_rate,
                 grid_size,
                 sample_ratio,
                 search_method='knn',
                 sample_method='grid'):
        self.kernel_size = kernel_size
        self.dilation_rate = [1 for _ in range(len(kernel_size))] if dilation_rate is None else dilation_rate
        self.grid_size = grid_size
        self.sample_ratio = sample_ratio
        self.radius = [5 * dl for dl in grid_size]
        self.search_method = search_method
        self.sample_method = sample_method

        self.transform = MultiScaleTransform(kernel_size=self.kernel_size,
                                             grid_size=self.grid_size,
                                             radius=self.radius,
                                             sample_ratio=self.sample_ratio,
                                             search_method=self.search_method,
                                             sample_method=self.sample_method)

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def set_transform(self, dataset):
        current_transform = getattr(dataset, "transform", None)
        if current_transform is None:
            setattr(dataset, "transform", self.transform)
        else:
            if (
                    isinstance(current_transform, Compose) and self.transform not in current_transform.transforms
            ):  # The transform contains several transformations
                current_transform.transforms += [self.transform]
            elif current_transform != self.transform:
                setattr(
                    dataset, "transform", Compose([current_transform, self.transform]),
                )

    # RandLA-Net style
    def _multiscale_compute_fn_dense(self,
                                     batch,
                                     collate_fn=None,
                                     precompute_multiscale=False):
        batch = collate_fn(batch)
        if not precompute_multiscale:
            return batch
        num_scales = len(self.kernel_size)
        multiscale = []
        pos = batch.pos     # [B, N, 3]
        for i in range(num_scales):
            if self.search_method.lower() == 'knn':
                neighbor_idx = batch_knn_search(pos, pos, self.kernel_size[i])      # [B, N, K]
            elif self.search_method.lower() == 'radius':
                neighbor_idx = batch_radius_search(pos, pos, self.radius[i], self.kernel_size[i])
            else:
                raise NotImplementedError('Only `knn` and `radius` search method are implemented!')
            sample_num = int(pos.shape[1] * self.sample_ratio[i])
            if self.sample_method.lower() == 'random':
                choice = torch.randperm(pos.shape[1])[:sample_num]
                sub_pos = pos[:, choice, :]             # random sampled pos   [B, S, 3]
                sub_idx = neighbor_idx[:, choice, :]    # the pool idx  [B, S, K]
            elif self.sample_method.lower() == 'fps':
                choice = furthest_point_sample(pos.cuda(), sample_num).to(torch.long).cpu()
                sub_pos = pos.gather(dim=1, index=choice.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
                sub_idx = neighbor_idx.gather(dim=1, index=choice.unsqueeze(-1).expand(-1, -1, neighbor_idx.shape[-1]))
            else:
                raise NotImplementedError('Only `random` and `fps` sampling method are implemented!')

            up_idx = batch_knn_search(sub_pos, pos, 1)      # [B, N, 1]
            multiscale.append(Data(pos=pos, neighbor_idx=neighbor_idx, sub_idx=sub_idx, up_idx=up_idx))
            pos = sub_pos

        data = MultiScaleData.from_data(batch)
        data.multiscale = multiscale
        return data

    # KPConv style
    def _multiscale_compute_fn_sparse(self,
                                      data,
                                      collate_fn=None,
                                      precompute_multiscale=False):
        data = collate_fn(data)
        if not precompute_multiscale:
            return data
        num_scales = len(self.kernel_size)
        multiscale = []
        pos, batch = data.pos, data.batch   # [N, 3]
        for i in range(num_scales):
            if self.search_method.lower() == 'knn':
                neighbor_idx = batch_knn_search(pos, pos, self.kernel_size[i] * self.dilation_rate[i], batch, batch)
                neighbor_idx = neighbor_idx[:, ::self.dilation_rate[i]]
            elif self.search_method.lower() == 'radius':
                neighbor_idx = batch_radius_search(pos, pos, self.radius[i], self.kernel_size[i], batch, batch)
            else:
                raise NotImplementedError('Only `knn` and `radius` search method are implemented!')

            if self.sample_method.lower() == 'random':
                _, batch_num = np.unique(batch, return_counts=True)
                batch_sample_num = np.round((batch_num * self.sample_ratio[i])).astype(np.int32)
                batch_cumsum = np.cumsum(batch_num)
                choice = None
                for (i, (num, sample_num)) in enumerate(zip(batch_num, batch_sample_num)):
                    batch_choice = torch.randperm(num)[:sample_num]
                    if i == 0:
                        choice = batch_choice
                    else:
                        batch_choice += batch_cumsum[i-1]
                        choice = torch.cat((choice, batch_choice), dim=0)
                sub_pos = pos[choice]
                sub_batch = batch[choice]
                sub_idx = neighbor_idx[choice]
            elif self.sample_method.lower() == 'fps':
                choice = fps(pos.cuda(), batch.cuda(), self.sample_ratio[i]).cpu()
                sub_pos = pos[choice]
                sub_batch = batch[choice]
                sub_idx = neighbor_idx[choice]
            elif self.sample_method.lower() == 'grid':
                cluster, indices = grid_sampling_sparse(pos, self.grid_size[i], batch=batch)
                sub_pos = scatter_mean(pos, cluster, dim=0)
                sub_batch = batch[indices]
                sub_idx = neighbor_idx[indices]
            else:
                raise NotImplementedError('Only `random`, `fps`, and `grid` sampling method are implemented!')

            up_idx = batch_knn_search(sub_pos, pos, self.kernel_size[i], sub_batch, batch)
            multiscale.append(Data(pos=pos, batch=batch, neighbor_idx=neighbor_idx, sub_idx=sub_idx, up_idx=up_idx))
            pos, batch = sub_pos, sub_batch

        ms_data = MultiScaleData.from_data(data)
        ms_data.multiscale = multiscale
        return ms_data

    def _dataloader_dense(self, dataset, precompute_multiscale, **kwargs):
        batch_collate_function = partial(self._multiscale_compute_fn_dense,
                                         collate_fn=SimpleBatch.from_data_list,
                                         precompute_multiscale=precompute_multiscale)
        data_loader_function = partial(DataLoader, collate_fn=batch_collate_function, worker_init_fn=np.random.seed)
        return data_loader_function(dataset, **kwargs)

    def _dataloader_sparse(self, dataset, precompute_multiscale, **kwargs):
        batch_collate_function = partial(self._multiscale_compute_fn_sparse,
                                         collate_fn=Batch.from_data_list,
                                         precompute_multiscale=precompute_multiscale)
        data_loader_function = partial(DataLoader, collate_fn=batch_collate_function, worker_init_fn=np.random.seed)
        return data_loader_function(dataset, **kwargs)

    def _dataloader_single(self, dataset, precompute_multiscale, **kwargs):
        if precompute_multiscale:
            self.set_transform(dataset)
        collate_fn = MultiScaleBatch.from_data_list
        data_loader_function = partial(DataLoader, collate_fn=collate_fn, worker_init_fn=np.random.seed)
        return data_loader_function(dataset, **kwargs)

    def create_dataloader(self,
                          batch_size: int,
                          precompute_multiscale: bool = True,
                          conv_type: str = 'dense'):
        if conv_type.lower() == 'dense':
            dataloader_fn = self._dataloader_dense
        elif conv_type.lower() == 'sparse':
            dataloader_fn = self._dataloader_sparse
        elif conv_type.lower() == 'single':
            dataloader_fn = self._dataloader_single
        else:
            raise NotImplementedError('Only `dense`, `sparse`, or `single` are implemented!')

        if self.train_set is not None:
            self.train_loader = dataloader_fn(self.train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              precompute_multiscale=precompute_multiscale)
        if self.val_set is not None:
            self.val_loader = dataloader_fn(self.val_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            precompute_multiscale=precompute_multiscale)
        if self.test_set is not None:
            self.test_loader = dataloader_fn(self.test_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             precompute_multiscale=precompute_multiscale)

