# -*- coding:utf-8 -*-
import os, sys, glob, pickle
from utils import read_ply, write_ply
import pandas as pd
from sklearn.neighbors import KDTree
from torch_geometric.data import Data, Dataset, InMemoryDataset
from .dataset import *

CLASS_NAMES = {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4, 'window': 5, 'door': 6,
               'table': 7, 'chair': 8, 'sofa': 9, 'bookcase': 10, 'board': 11, 'clutter': 12}


class S3DIS(InMemoryDataset):
    data_dir = 'Stanford3dDataset_v1.2_Aligned_Version'

    def __init__(self,
                 root,
                 test_area=5,
                 grid_size=0.04,
                 num_points=65536,
                 radius=5.0,
                 sample_per_epoch=100,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        assert test_area in [1, 2, 3, 4, 5, 6]
        super(S3DIS, self).__init__(root, transform, pre_transform, pre_filter)
        self.test_area = 'Area_{}'.format(test_area)
        self.grid_size = grid_size
        self.num_points = num_points
        self.radius = radius
        self.sample_per_epoch = sample_per_epoch
        self.train = train

        self.label_values = np.sort([v for k, v in CLASS_NAMES.items()])        # [0-12]

        self.possibility = []
        self.min_possibility = []
        self.input_trees = []
        self.input_rgb = []
        self.input_labels = []
        self.input_names = []

        if not self.train:
            self.val_proj = []
            self.val_labels = []

        # load processed data
        self._load_processed()

        # random init probability
        for tree in self.input_trees:
            self.possibility += [np.random.randn(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

    @property
    def raw_file_names(self):
        return ['Area_1_anno.txt', 'Area_2_anno.txt', 'Area_3_anno.txt',
                'Area_4_anno.txt', 'Area_5_anno.txt', 'Area_6_anno.txt']

    @property
    def processed_file_names(self):
        return ['original', 'sampled']

    def __len__(self):
        if self.sample_per_epoch > 0:
            return self.sample_per_epoch
        else:
            return len(self.input_trees)

    def get(self, idx):
        return self._get_random()

    def download(self):
        pass

    def process(self):
        # Note: there is an extra character in line 180389 of Area_5/hallway_6/Annotations/ceiling_1.txt
        for path in self.processed_paths:
            os.makedirs(path)
        for i, path in enumerate(self.raw_paths):
            print("Processing Area_{}...".format(i + 1))
            anno_paths = [line.rstrip() for line in open(path)]
            anno_paths = [os.path.join(self.raw_dir, self.data_dir, p) for p in anno_paths]
            for anno_path in anno_paths:
                print('Processing {}...'.format(anno_path))
                elements = anno_path.split('/')
                filename = elements[-3] + '_' + elements[-2]
                data_list = []
                for f in glob.glob(os.path.join(anno_path, '*.txt')):
                    print('Collecting {}...'.format(f))
                    label = os.path.basename(f).split('_')[0]
                    if label not in CLASS_NAMES:
                        label = 'clutter'
                    # cls_points = np.loadtxt(f)
                    cls_points = pd.read_csv(f, header=None, delim_whitespace=True).values  # pandas for faster reading
                    cls_labels = np.full((cls_points.shape[0], 1), CLASS_NAMES[label], dtype=np.int32)
                    data_list.append(np.concatenate([cls_points, cls_labels], axis=1))  # Nx7

                points_labels = np.concatenate(data_list, axis=0)

                xyz_min = np.amin(points_labels, axis=0)[0:3]
                points_labels[:, 0:3] -= xyz_min    # aligned to the minimal point
                xyz = points_labels[:, 0:3].astype(np.float32)
                rgb = points_labels[:, 3:6].astype(np.uint8)
                labels = points_labels[:, 6].astype(np.uint8)

                org_ply_file = os.path.join(self.processed_paths[0], filename + '.ply')
                write_ply(org_ply_file, [xyz, rgb, labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                # save sub_cloud and KDTree files
                sub_xyz, sub_rgb, sub_labels = grid_sub_sampling(xyz, rgb, labels, self.grid_size)
                sub_rgb = sub_rgb / 255.

                sub_ply_file = os.path.join(self.processed_paths[1], filename + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                search_tree = KDTree(sub_xyz)
                kd_tree_file = os.path.join(self.processed_paths[1], filename + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], filename + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

    def _load_processed(self):
        for f in glob.glob(os.path.join(self.processed_paths[0], '*.ply')):
            name = f.split('/')[-1][:-4]
            if self.train:
                if self.test_area in name:
                    continue
            else:
                if self.test_area not in name:
                    continue

            kd_tree_file = os.path.join(self.processed_paths[1], '{}_KDTree.pkl'.format(name))
            sub_ply_file = os.path.join(self.processed_paths[1], '{}.ply'.format(name))
            data = read_ply(sub_ply_file)
            sub_rgb = np.vstack((data['r'], data['g'], data['b'])).T
            sub_labels = data['class']
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_rgb += [sub_rgb]
            self.input_labels += [sub_labels]
            self.input_names += [name]

            if not self.train:
                proj_file = os.path.join(self.processed_paths[1], '{}_proj.pkl'.format(name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

    def _get_random(self):
        cloud_idx = int(np.argmin(self.min_possibility))
        pick_idx = np.argmin(self.possibility[cloud_idx])
        points = np.array(self.input_trees[cloud_idx].data, copy=False)
        pick_point = points[pick_idx, :].reshape(1, -1)

        noise = np.random.normal(scale=3.5 / 10, size=pick_point.shape)
        pick_point = pick_point + noise.astype(pick_point.dtype)

        if len(points) < self.num_points:
            query_idx = self.input_trees[cloud_idx].query(pick_point, k=len(points), return_distance=False)[0]
        else:
            query_idx = self.input_trees[cloud_idx].query(pick_point, k=self.num_points, return_distance=False)[0]

        # query_idx = self.input_trees[cloud_idx].query_radius(pick_point, r=self.radius, return_distance=False)[0]

        np.random.shuffle(query_idx)
        query_xyz = points[query_idx] - pick_point
        query_rgb = self.input_rgb[cloud_idx][query_idx]
        query_labels = self.input_labels[cloud_idx][query_idx]

        # update possibility, reduce the posibility of chosen cloud and point
        dists = np.sum(np.square(query_xyz.astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[cloud_idx][query_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        pos = torch.from_numpy(query_xyz).to(torch.float32)
        rgb = torch.from_numpy(query_rgb).to(torch.float32)
        labels = torch.from_numpy(query_labels).to(torch.long)
        point_idx = torch.from_numpy(query_idx).to(torch.long)
        # cloud_idx = torch.Tensor([cloud_idx]).to(torch.long)
        cloud_idx = torch.tensor([cloud_idx]).to(torch.long)
        data = Data(pos=pos, rgb=rgb, y=labels, point_idx=point_idx, cloud_idx=cloud_idx)

        # upsampled with minimal replacement
        # if len(points) < self.num_points:
        #     data = FixedPoints(self.num_points, replace=False, allow_duplicates=True)(data)

        return data


class S3DISDataset(BaseDataset):
    def __init__(self,
                 root,
                 test_area=5,
                 first_grid_size=0.04,
                 num_points=40960,
                 radius=2.0,
                 kernel_size=None,
                 dilation_rate=None,
                 grid_size=None,
                 sample_ratio=None,
                 search_method=None,
                 sample_method=None,
                 train_sample_per_epoch=500,
                 test_sample_per_epoch=100,
                 train_transform=None,
                 test_transform=None,
                 pre_transform=None):
        super(S3DISDataset, self).__init__(kernel_size,
                                           dilation_rate,
                                           grid_size,
                                           sample_ratio,
                                           search_method,
                                           sample_method)

        self.train_set = S3DIS(root,
                               test_area=test_area,
                               grid_size=first_grid_size,
                               num_points=num_points,
                               radius=radius,
                               sample_per_epoch=train_sample_per_epoch,
                               train=True,
                               transform=train_transform,
                               pre_transform=pre_transform)

        self.test_set = S3DIS(root,
                              test_area=test_area,
                              grid_size=first_grid_size,
                              num_points=num_points,
                              radius=radius,
                              sample_per_epoch=test_sample_per_epoch,
                              train=False,
                              transform=test_transform,
                              pre_transform=pre_transform)

        self.val_set = self.test_set


if __name__ == '__main__':
    root = '/media/yangfei/HGST3/DATA/S3DISRoom'
    dataset = S3DISDataset(root, test_area=5, num_points=40960,
                           kernel_size=[16, 16, 16, 16, 16],
                           grid_size=[0.08, 0.16, 0.32, 0.64, 1.28],
                           sample_ratio=[0.25, 0.25, 0.25, 0.25, 0.5],
                           search_method='radius',
                           sample_method='grid')

    dataset.create_dataloader(batch_size=8, precompute_multiscale=True, conv_type='sparse')
    for data in dataset.train_loader:
        print(data)
    print(dataset)


