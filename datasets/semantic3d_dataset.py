# -*- coding:utf-8 -*-
import os, sys, glob, pickle
import pandas as pd
from sklearn.neighbors import KDTree
from torch_geometric.data import Data, Dataset, InMemoryDataset
from utils import read_ply, write_ply
from utils import Plot
from .dataset import *


CLASS_NAMES = {'unlabeled': 0,
               'man-made terrain': 1,
               'natural terrain': 2,
               'high vegetation': 3,
               'low vegetation': 4,
               'buildings': 5,
               'hard scape': 6,
               'scanning artefacts': 7,
               'cars': 8}


class Semantic3D(InMemoryDataset):
    def __init__(self,
                 root,
                 split='train',
                 grid_size=0.06,
                 num_points=65536,
                 radius=5.0,
                 sample_per_epoch=100,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        assert split in ['train', 'val', 'test']
        self.grid_size = grid_size
        self.num_points = num_points
        self.radius = radius
        self.sample_per_epoch = sample_per_epoch
        self.split = split
        self.label_values = np.sort([v for k, v in CLASS_NAMES.items()])        # [0-8], 0 for unlabeled
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignore_labels = np.argsort([0])
        super(Semantic3D, self).__init__(root, transform, pre_transform, pre_filter)

        # ===================================================== split ===========================================
        # Following KPConv and RandLA-Net train-val split
        all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        val_split = 1
        # Initial train-val-test files
        train_files = []
        val_files = []
        test_files = []
        cloud_names = [filename[:-4] for filename in os.listdir(self.raw_paths[0]) if filename[-4:] == '.txt']
        for cloud_name in cloud_names:
            if os.path.exists(os.path.join(self.raw_paths[0], cloud_name + '.labels')):
                train_files.append(os.path.join(self.processed_paths[1], cloud_name + '.ply'))
            else:
                test_files.append(os.path.join(self.processed_paths[0], cloud_name + '.ply'))
        train_files = np.sort(train_files)
        test_files = np.sort(test_files)

        for i, filename in enumerate(train_files):
            if all_splits[i] == val_split:
                val_files.append(filename)

        train_files = np.sort([x for x in train_files if x not in val_files])

        if split is 'train':
            self.file_list = train_files
        elif self.split == 'val':
            self.file_list = val_files
        elif self.split == 'test':
            self.file_list = test_files
        else:
            raise ValueError('Only `train`, `val` or `test` split is supported !')
        # ===================================================== split ===========================================

        # Initial containers
        self.possibility = []
        self.min_possibility = []
        self.class_weight = []
        self.input_trees = []
        self.input_rgb = []
        self.input_labels = []

        if split in ['val', 'test']:
            self.test_proj = []
            self.test_labels = []

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        # load processed data
        self._load_processed()

        # random init probability
        for tree in self.input_trees:
            self.possibility += [np.random.randn(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

        if split is not 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels), return_counts=True)
            self.class_weight += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

    @property
    def raw_file_names(self):
        return ['txt']

    @property
    def processed_file_names(self):
        return ['original_reduced', 'sampled']

    def __len__(self):
        if self.sample_per_epoch > 0:
            return self.sample_per_epoch
        else:
            return len(self.input_trees)

    def get(self, idx):
        return self._get_random()

    def download(self):
        pass

    @staticmethod
    def _load_cloud(filename):
        pc = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float32)
        pc = pc.values
        return pc

    @staticmethod
    def _load_label(filename):
        label = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        label = label.values
        return label

    def process(self):
        for path in self.processed_paths:
            os.makedirs(path)
        for pc_path in glob.glob(os.path.join(self.raw_paths[0], '*.txt')):
            print('Processing {} ...'.format(pc_path))
            cloud_name = pc_path.split('/')[-1][:-4]
            if os.path.exists(os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')):
                continue
            pc = self._load_cloud(pc_path)
            label_path = pc_path[:-4] + '.labels'
            if os.path.exists(label_path):
                labels = self._load_label(label_path)
                org_ply_path = os.path.join(self.processed_paths[0], cloud_name + '.ply')
                # Subsample the training set cloud to the same resolution 0.01 as the test set
                xyz, rgb, labels = grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                     pc[:, 4:7].astype(np.uint8),
                                                     labels, grid_size=0.01)
                labels = np.squeeze(labels)
                # save sub-sampled original cloud
                write_ply(org_ply_path, [xyz, rgb, labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                # save sub_cloud and KDTree file
                sub_xyz, sub_rgb, sub_labels = grid_sub_sampling(xyz, rgb, labels, grid_size=self.grid_size)
                sub_rgb = sub_rgb / 255.
                sub_labels = np.squeeze(sub_labels)
                sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                search_tree = KDTree(sub_xyz, leaf_size=50)
                kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

            else:
                org_ply_path = os.path.join(self.processed_paths[0], cloud_name + '.ply')
                write_ply(org_ply_path, [pc[:, :3].astype(np.float32), pc[:, 4:7].astype(np.uint8)],
                          ['x', 'y', 'z', 'r', 'g', 'b'])

                sub_xyz, sub_rgb = grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                     pc[:, 4:7].astype(np.uint8),
                                                     grid_size=self.grid_size)

                sub_rgb = sub_rgb / 255.
                sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb], ['x', 'y', 'z', 'r', 'g', 'b'])

                search_tree = KDTree(sub_xyz, leaf_size=50)
                kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                labels = np.zeros(pc.shape[0], dtype=np.uint8)
                proj_idx = np.squeeze(search_tree.query(pc[:, :3].astype(np.float32), return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

    def _load_processed(self):
        for i, filename in enumerate(self.file_list):
            cloud_name = filename.split('/')[-1][:-4]
            print('Load cloud {:d}: {:s}'.format(i, cloud_name))
            kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
            sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
            # read ply data
            data = read_ply(sub_ply_file)
            sub_rgb = np.vstack((data['r'], data['g'], data['b'])).T
            if self.split is 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_rgb += [sub_rgb]
            if self.split in ['train', 'val']:
                self.input_labels += [sub_labels]

            if self.split in ['val', 'test']:
                print('Preparing re-projection indices for val and test')
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                    self.test_proj += [proj_idx]
                    self.test_labels += [labels]

        print('Finished.')

    def _get_random(self):
        cloud_idx = int(np.argmin(self.min_possibility))
        pick_idx = np.argmin(self.possibility[cloud_idx])
        points = np.array(self.input_trees[cloud_idx].data, copy=False)
        pick_point = points[pick_idx, :].reshape(1, -1)

        noise = np.random.normal(scale=3.5 / 10, size=pick_point.shape)
        pick_point = pick_point + noise.astype(pick_point.dtype)

        # Semantic3D is a big dataset with large cloud, so there is no need to resample
        query_idx = self.input_trees[cloud_idx].query(pick_point, k=self.num_points, return_distance=False)[0]
        # query_idx = self.input_trees[cloud_idx].query_radius(pick_point, r=self.radius, return_distance=False)[0]

        np.random.shuffle(query_idx)
        query_xyz = points[query_idx]
        query_xyz[:, 0:2] = query_xyz[:, 0:2] - pick_point[:, 0:2]      # centerize in xOy plane
        query_rgb = self.input_rgb[cloud_idx][query_idx]
        if self.split is 'test':
            query_labels = np.zeros(query_xyz.shape[0])
            query_weights = 1
        else:
            query_labels = self.input_labels[cloud_idx][query_idx]
            query_labels = np.array([self.label_to_idx[l] for l in query_labels])
            query_weights = np.array([self.class_weight[0][n] for n in query_labels])

        # update possibility, reduce the possibility of chosen cloud and point
        dists = np.sum(np.square(points[query_idx] - pick_point).astype(np.float32), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * query_weights
        self.possibility[cloud_idx][query_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        pos = torch.from_numpy(query_xyz).to(torch.float32)
        rgb = torch.from_numpy(query_rgb).to(torch.float32)
        labels = torch.from_numpy(query_labels).to(torch.long)
        point_idx = torch.from_numpy(query_idx).to(torch.long)
        cloud_idx = torch.tensor([cloud_idx]).to(torch.long)
        data = Data(pos=pos, rgb=rgb, y=labels - 1, point_idx=point_idx, cloud_idx=cloud_idx)   # ignore label 0

        return data


class Semantic3DDataset(BaseDataset):
    def __init__(self,
                 root,
                 first_grid_size=0.06,
                 num_points=65536,
                 radius=5.0,
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
        super(Semantic3DDataset, self).__init__(kernel_size,
                                                dilation_rate,
                                                grid_size,
                                                sample_ratio,
                                                search_method,
                                                sample_method)

        self.train_set = Semantic3D(root,
                                    split='train',
                                    grid_size=first_grid_size,
                                    num_points=num_points,
                                    radius=radius,
                                    sample_per_epoch=train_sample_per_epoch,
                                    transform=train_transform,
                                    pre_transform=pre_transform)

        self.val_set = Semantic3D(root,
                                  split='val',
                                  grid_size=first_grid_size,
                                  num_points=num_points,
                                  radius=radius,
                                  sample_per_epoch=test_sample_per_epoch,
                                  transform=test_transform,
                                  pre_transform=pre_transform)

        self.test_set = Semantic3D(root,
                                   split='test',
                                   grid_size=first_grid_size,
                                   num_points=num_points,
                                   radius=radius,
                                   sample_per_epoch=test_sample_per_epoch,
                                   transform=test_transform,
                                   pre_transform=pre_transform)


if __name__ == '__main__':
    root = '/media/yangfei/HGST3/DATA/Semantic3D'
    dataset = Semantic3DDataset(root=root, first_grid_size=0.06, num_points=65536,
                                kernel_size=[16, 16, 16, 16, 16],
                                dilation_rate=[1, 1, 1, 1, 1],
                                grid_size=[0.12, 0.24, 0.48, 0.96, 1.92],
                                sample_ratio=[0.25, 0.25, 0.25, 0.25, 0.5],
                                search_method='knn', sample_method='grid')
    for data in dataset.train_set:
        Plot.draw_pc(torch.cat([data.pos, data.rgb], dim=-1).numpy())
        Plot.draw_pc_sem_ins(data.pos.numpy(), data.y.numpy())
    dataset.create_dataloader(batch_size=8, precompute_multiscale=True, conv_type='sparse')
    for data in dataset.train_loader:
        print(data)
        batch = 0
        Plot.draw_pc(data.multiscale[0].pos[data.multiscale[0].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[1].pos[data.multiscale[1].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[2].pos[data.multiscale[2].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[3].pos[data.multiscale[3].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[4].pos[data.multiscale[4].batch == batch].numpy())
    print(dataset)

