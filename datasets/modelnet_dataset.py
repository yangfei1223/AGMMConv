import re
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints, RandomRotate
from torch_points3d.datasets.classification.modelnet import ModelNet, SampledModelNet
from utils import Plot
from .dataset import *


class UniformSampling(object):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __call__(self, data):
        num_nodes = data.num_nodes
        pos = data.pos.unsqueeze(0)
        assert self.sample_num <= num_nodes

        choice = furthest_point_sample(pos.cuda(), self.sample_num).to(torch.long).cpu().squeeze()
        # sub_pos = pos.gather(dim=1, index=choice.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
        # sub_idx = neighbor_idx.gather(dim=1, index=choice.unsqueeze(-1).expand(-1, -1, neighbor_idx.shape[-1]))

        for key, item in data:
            if bool(re.search('edge', key)):
                continue
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.sample_num)


class ModelNetDataset(BaseDataset):
    def __init__(self,
                 root,
                 category='40',
                 kernel_size=None,
                 dilation_rate=None,
                 grid_size=None,
                 sample_ratio=None,
                 search_method=None,
                 sample_method=None,
                 train_transform=None,
                 test_transform=None,
                 pre_transform=None):
        super(ModelNetDataset, self).__init__(kernel_size,
                                              dilation_rate,
                                              grid_size,
                                              sample_ratio,
                                              search_method,
                                              sample_method)

        self.train_set = SampledModelNet(root,
                                         name=category,
                                         train=True,
                                         transform=train_transform,
                                         pre_transform=pre_transform)
        self.test_set = SampledModelNet(root,
                                        name=category,
                                        train=False,
                                        transform=test_transform,
                                        pre_transform=pre_transform)
        self.val_set = self.test_set

    def __repr__(self):
        return '{}, train_set size: {}, val_set size: {}, test_set size: {}'.format(self.__class__,
                                                                                    len(self.train_set),
                                                                                    len(self.val_set),
                                                                                    len(self.test_set))


if __name__ == '__main__':
    root = '../DATA/ModelNetSampled'
    pre_transform = Compose([
        NormalizeScale(),
        GridSampling3D(size=0.02, mode='last')
    ])
    transform = FixedPoints(1024)
    # dataset = SampledModelNet(root, name='40', train=True, transform=transform, pre_transform=pre_transform)
    # print(dataset)
    # for data in dataset:
    #     Plot.draw_pc(data.pos.numpy())
    #     # Plot.draw_pc_sem_ins(data.pos.numpy(), data.y.numpy())
    dataset = ModelNetDataset(root=root,
                              category='40',
                              kernel_size=[32, 16, 16, 16, 16],
                              grid_size=[0.04, 0.08, 0.16, 0.32, 0.64],
                              sample_ratio=[0.5, 0.5, 0.5, 0.5, 0.5],
                              search_method='knn',
                              sample_method='fps',
                              train_transform=transform,
                              test_transform=transform,
                              pre_transform=pre_transform)
    dataset.create_dataloader(batch_size=16, precompute_multiscale=True, conv_type='sparse')
    for data in dataset.train_loader:
        print(data)
        batch = 0
        Plot.draw_pc(data.multiscale[0].pos[data.multiscale[0].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[1].pos[data.multiscale[1].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[2].pos[data.multiscale[2].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[3].pos[data.multiscale[3].batch == batch].numpy())
        Plot.draw_pc(data.multiscale[4].pos[data.multiscale[4].batch == batch].numpy())
    print(dataset)

