from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints, RandomRotate
from torch_points3d.core.data_transform import *
import datasets, models


def get_class_weights(dataset):
    # pre-calculate the number of points in each category
    num_per_class = []
    if dataset is 'S3DIS':
        num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                  650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    elif dataset is 'Semantic3D':
        num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                 dtype=np.int32)
    else:
        raise ValueError('Unsupported dataset!')
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    return torch.from_numpy(ce_label_weight.astype(np.float32))

class Config(object):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 task=None
                 ):
        self.device = 'cuda'
        self.device_id = device_id
        self.mode = mode
        self.batch_size = batch_size
        self.task = task
        if self.task == 'object':
            self.epochs = 250
            self.lr = 1e-3
            self.gamma = 0.1 ** 0.01
            self.metric = 'Overall Acc'
        elif self.task == 'scene':
            self.epochs = 100
            self.lr = 1e-2
            self.gamma = 0.1 ** 0.02
            self.metric = 'Mean IoU'
        else:
            raise ValueError('Please specify task type in [classification, segmentation]')
        self.momentum = 0.98
        self.weight_decay = 1e-4

class ModelNetConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=16,
                 category='40',
                 num_kernels=8,
                 test_area=None,
                 task='object'
                 ):
        super(ModelNetConfig, self).__init__(device_id, mode, batch_size, task)
        self.root = 'DATA/ModelNetSampled'
        self.dataset_name = 'ModelNetDataset'
        self.category = category
        self.first_grid_size = 0.02
        self.sample_num = 1024
        self.num_classes = int(category)
        self.ignore_index = -1
        self.class_weights = None
        self.model_name = 'ResidualGMMNet'
        self.num_kernels = num_kernels
        self.is_pooling = False
        self.layers = [32, 64, 128, 256, 512]
        self.kernel_size = [16, 16, 16, 16, 16]
        self.dilation_rate = [1, 1, 1, 1, 1]
        self.grid_size = [0.04, 0.08, 0.16, 0.32, 0.64]
        self.sample_ratio = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.search_method = 'knn'
        self.sample_method = 'fps'
        self.conv_type = 'sparse'
        self.prefix = '{}_on_{}{}'.format(self.model_name, self.dataset_name, self.category)
        self.model_path = 'RUNS/checkpoints/{}_sn_1024_bs_16_nk_{}.ckpt'.format(self.prefix, num_kernels)
        self.save_path = 'RUNS/results/{}{}'.format(self.dataset_name, self.category)

        self.pre_transform = Compose([
            NormalizeScale(),
            GridSampling3D(size=self.first_grid_size, mode='last')
        ])
        self.train_transform = Compose([
            FixedPoints(num=self.sample_num, replace=False, allow_duplicates=True),
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'norm'],
                           delete_feats=[False, True]),
            RandomScaleAnisotropic(scales=[0.8, 1.2], anisotropic=True),
            RandomSymmetry(axis=[True, True, True]),
            RandomRotate(degrees=180, axis=-1),
            RandomNoise(sigma=0.001),
        ])
        self.test_transform = Compose([
            FixedPoints(num=self.sample_num, replace=False, allow_duplicates=True),
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'norm'],
                           delete_feats=[False, True]),
        ])
        self.dataset_fn = partial(getattr(datasets, self.dataset_name),
                                  root=self.root,
                                  category=self.category,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=self.dilation_rate,
                                  grid_size=self.grid_size,
                                  sample_ratio=self.sample_ratio,
                                  search_method=self.search_method,
                                  sample_method=self.sample_method,
                                  train_transform=self.train_transform,
                                  test_transform=self.test_transform,
                                  pre_transform=self.pre_transform)

        self.model_fn = partial(getattr(models, self.model_name),
                                in_channels=6,
                                n_classes=self.num_classes,
                                num_kernels=self.num_kernels,
                                layers=self.layers,
                                class_weights=self.class_weights,
                                ignore_index=self.ignore_index,
                                is_pooling=self.is_pooling)

class ModelNetNormalConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=16,
                 category='40',
                 num_kernels=8,
                 task='object'
                 ):
        super(ModelNetNormalConfig, self).__init__(device_id, mode, batch_size, task)
        self.root = 'DATA/ModelNetSampled'
        self.dataset_name = 'ModelNetDataset'
        self.category = category
        self.first_grid_size = 0.02
        self.sample_num = 1024
        self.num_classes = int(category)
        self.model_name = 'ResudualGMMNormalEstimation'
        self.num_kernels = num_kernels
        self.is_transpose = True
        self.layers = [32, 64, 128, 256, 512]
        self.kernel_size = [16, 16, 16, 16, 16]
        self.dilation_rate = [1, 1, 1, 1, 1]
        self.grid_size = [0.04, 0.08, 0.16, 0.32, 0.64]
        self.sample_ratio = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.search_method = 'knn'
        self.sample_method = 'fps'
        self.conv_type = 'sparse'
        self.prefix = '{}_on_{}{}'.format(self.model_name, self.dataset_name, self.category)
        self.model_path = 'RUNS/checkpoints/{}_sn_1024_bs_16.ckpt'.format(self.prefix)
        self.save_path = 'RUNS/results/{}{}'.format(self.dataset_name, self.category)

        self.epochs = 250
        self.lr = 1e-2
        self.gamma = 0.1 ** 0.01

        self.pre_transform = Compose([
            NormalizeScale(),
            GridSampling3D(size=self.first_grid_size, mode='last')
        ])
        self.train_transform = Compose([
            FixedPoints(num=self.sample_num, replace=False, allow_duplicates=True),
            AddFeatsByKeys(list_add_to_x=[True],
                           feat_names=['pos'],
                           delete_feats=[False]),
            RandomScaleAnisotropic(scales=[0.8, 1.2], anisotropic=True),
            RandomSymmetry(axis=[True, True, True]),
            RandomRotate(degrees=180, axis=-1),
            RandomNoise(sigma=0.001),
        ])
        self.test_transform = Compose([
            FixedPoints(num=self.sample_num, replace=False, allow_duplicates=True),
            AddFeatsByKeys(list_add_to_x=[True],
                           feat_names=['pos'],
                           delete_feats=[False]),
        ])
        self.dataset_fn = partial(getattr(datasets, self.dataset_name),
                                  root=self.root,
                                  category=self.category,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=self.dilation_rate,
                                  grid_size=self.grid_size,
                                  sample_ratio=self.sample_ratio,
                                  search_method=self.search_method,
                                  sample_method=self.sample_method,
                                  train_transform=self.train_transform,
                                  test_transform=self.test_transform,
                                  pre_transform=self.pre_transform)

        self.model_fn = partial(getattr(models, self.model_name),
                                in_channels=3,
                                num_kernels=self.num_kernels,
                                layers=self.layers,
                                is_transpose=self.is_transpose)

class S3DISConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 num_kernels=8,
                 category=None,
                 test_area=5,
                 task='scene'
                 ):
        super(S3DISConfig, self).__init__(device_id, mode, batch_size, task)
        self.root = 'DATA/S3DISRoom'
        self.dataset_name = 'S3DISDataset'
        self.test_area = test_area
        self.first_grid_size = 0.04
        self.sample_num = 40960
        self.sample_radius = self.first_grid_size * 50
        self.num_classes = 13
        self.ignore_index = -1
        self.class_weights = get_class_weights('S3DIS')
        self.model_name = 'ResidualGMMSegNet'
        self.num_kernels = num_kernels
        self.is_transpose = True
        self.layers = [32, 64, 128, 256, 512]
        self.kernel_size = [16, 16, 16, 16, 16]
        self.dilation_rate = [1, 1, 1, 1, 1]
        self.grid_size = [0.08, 0.16, 0.32, 0.64, 1.28]
        self.sample_ratio = [0.25, 0.25, 0.25, 0.25, 0.5]
        self.search_method = 'knn'
        self.sample_method = 'grid'
        self.conv_type = 'sparse'
        self.train_steps = 500
        self.test_steps = 100
        self.prefix = '{}_on_{}_Area_{}'.format(self.model_name, self.dataset_name, self.test_area)
        self.model_path = 'RUNS/checkpoints/{}.ckpt'.format(self.prefix)
        self.save_path = 'RUNS/results/{}/Area_{}'.format(self.dataset_name, self.test_area)

        self.train_transform = Compose([
            RandomScaleAnisotropic(scales=[0.8, 1.2]),
            RandomSymmetry(axis=[True, False, False]),
            RandomRotate(degrees=180, axis=-1),
            RandomNoise(sigma=0.001),
            DropFeature(drop_proba=0.2, feature_name='rgb'),
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'rgb'],
                           delete_feats=[False, True])
        ])
        self.test_transform = Compose([
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'rgb'],
                           delete_feats=[False, True])
        ])
        self.dataset_fn = partial(getattr(datasets, self.dataset_name),
                                  root=self.root,
                                  test_area=self.test_area,
                                  first_grid_size=self.first_grid_size,
                                  num_points=self.sample_num,
                                  radius=self.sample_radius,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=self.dilation_rate,
                                  grid_size=self.grid_size,
                                  sample_ratio=self.sample_ratio,
                                  search_method=self.search_method,
                                  sample_method=self.sample_method,
                                  train_sample_per_epoch=self.batch_size * self.train_steps,
                                  test_sample_per_epoch=self.batch_size * self.test_steps,
                                  train_transform=self.train_transform,
                                  test_transform=self.test_transform)

        self.model_fn = partial(getattr(models, self.model_name),
                                in_channels=6,
                                n_classes=self.num_classes,
                                num_kernels=self.num_kernels,
                                layers=self.layers,
                                class_weights=self.class_weights,
                                ignore_index=self.ignore_index,
                                is_transpose=self.is_transpose)

class Semantic3DConfig(Config):
    def __init__(self,
                 device_id=0,
                 mode='train',
                 batch_size=8,
                 num_kernels=8,
                 category=None,
                 test_area=None,     # make it compatible to S3DIS dataset
                 task='scene'
                 ):
        super(Semantic3DConfig, self).__init__(device_id, mode, batch_size, task)
        self.root = 'DATA/Semantic3D'
        self.dataset_name = 'Semantic3DDataset'
        self.first_grid_size = 0.06
        self.sample_num = 65536
        self.sample_radius = self.first_grid_size * 50
        self.num_classes = num_kernels
        self.ignore_index = -1
        self.class_weights = get_class_weights('Semantic3D')
        self.model_name = 'ResidualGMMSegNet'
        self.num_kernels = 8
        self.is_transpose = True
        self.layers = [32, 64, 128, 256, 512]
        self.kernel_size = [16, 16, 16, 16, 16]
        self.dilation_rate = [1, 1, 1, 1, 1]
        self.grid_size = [0.12, 0.24, 0.48, 0.96, 1.92]
        self.sample_ratio = [0.25, 0.25, 0.25, 0.25, 0.5]
        self.search_method = 'knn'
        self.sample_method = 'grid'
        self.conv_type = 'sparse'
        self.train_steps = 500
        self.test_steps = 100
        self.prefix = '{}_on_{}'.format(self.model_name, self.dataset_name)
        self.model_path = 'RUNS/checkpoints/{}_sn_65536_bs_4.ckpt'.format(self.prefix)
        self.save_path = 'RUNS/{}'.format(self.dataset_name)

        self.train_transform = Compose([RandomScaleAnisotropic(scales=[0.8, 1.2], anisotropic=True),
                                        RandomSymmetry(axis=[True, False, False]),
                                        RandomRotate(degrees=180, axis=-1),
                                        RandomNoise(sigma=0.001),
                                        DropFeature(drop_proba=0.2, feature_name='rgb'),
                                        AddFeatsByKeys(list_add_to_x=[True, True],
                                                       feat_names=['pos', 'rgb'],
                                                       delete_feats=[False, True])
                                        ])
        self.test_transform = Compose([AddFeatsByKeys(list_add_to_x=[True, True],
                                                      feat_names=['pos', 'rgb'],
                                                      delete_feats=[False, True])
                                       ])

        self.dataset_fn = partial(getattr(datasets, self.dataset_name),
                                  root=self.root,
                                  first_grid_size=self.first_grid_size,
                                  num_points=self.sample_num,
                                  radius=self.sample_radius,
                                  kernel_size=self.kernel_size,
                                  dilation_rate=self.dilation_rate,
                                  grid_size=self.grid_size,
                                  sample_ratio=self.sample_ratio,
                                  search_method=self.search_method,
                                  sample_method=self.sample_method,
                                  train_sample_per_epoch=self.batch_size * self.train_steps,
                                  test_sample_per_epoch=self.batch_size * self.test_steps,
                                  train_transform=self.train_transform,
                                  test_transform=self.test_transform)

        self.model_fn = partial(getattr(models, self.model_name),
                                in_channels=6,
                                n_classes=self.num_classes,
                                num_kernels=self.num_kernels,
                                layers=self.layers,
                                class_weights=self.class_weights,
                                ignore_index=self.ignore_index,
                                is_transpose=self.is_transpose)