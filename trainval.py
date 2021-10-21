# -*- coding:utf-8 -*-
'''
@Time : 2021/2/19 下午1:06
@Author: yangfei
@File : trainval.py
'''
import os, sys, time, pickle, argparse
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from utils import iou_from_confusions, runningScore, read_ply, write_ply
import configure

logging.getLogger().setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.cuda.set_device(cfg.device_id)

        self.dataset = cfg.dataset_fn()
        self.dataset.create_dataloader(batch_size=cfg.batch_size, conv_type=cfg.conv_type)
        self.model = cfg.model_fn()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=cfg.lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.gamma)

        self.metrics = runningScore(cfg.num_classes, ignore_index=cfg.ignore_index)

        self.test_probs = None

    def train_one_epoch(self, epoch):
        self.model.train()
        self.metrics.reset()
        with Ctq(self.dataset.train_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Train epoch[{}]'.format(epoch))
                data = data.to(self.device)
                self.optimizer.zero_grad()
                self.model.forward(data)
                self.model.compute_loss()
                self.model.backward()
                self.optimizer.step()
                tq_loader.set_postfix(internal_loss=self.model.internal_loss.item(),
                                      external_loss=self.model.external_loss.item(),
                                      total_loss=self.model.loss.item())
                self.metrics.update(self.model.y.cpu().numpy(), self.model.y_hat.max(dim=1)[1].cpu().numpy())

    def val_one_epoch(self, epoch):
        self.model.eval()
        self.metrics.reset()
        with Ctq(self.dataset.val_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Val epoch[{}]'.format(epoch))
                data = data.to(self.device)
                with torch.no_grad():
                    self.model.forward(data)
                self.model.compute_loss()
                tq_loader.set_postfix(internal_loss=self.model.internal_loss.item(),
                                      external_loss=self.model.external_loss.item(),
                                      total_loss=self.model.loss.item())
                self.metrics.update(self.model.y.cpu().numpy(), self.model.y_hat.max(dim=1)[1].cpu().numpy())

    def trainval(self):
        best_metric = 0
        # track parameters
        self.model.to(self.device)
        for epoch in range(self.cfg.epochs):
            logging.info('Training epoch: {}, learning rate: {}'.format(epoch, self.scheduler.get_last_lr()[0]))

            # training
            self.train_one_epoch(epoch)
            score_dict, _ = self.metrics.get_scores()
            logging.info('Training OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100, score_dict['Mean IoU'] * 100))
            # validation
            self.val_one_epoch(epoch)
            score_dict, _ = self.metrics.get_scores()
            logging.info('Test OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100, score_dict['Mean IoU'] * 100))

            current_metric = score_dict[self.cfg.metric]
            if best_metric <= current_metric:
                best_metric = current_metric
                self.model.save(self.cfg.model_path)     # save model
                logging.info('Save {} succeed, best {}: {:.2f} % !'.format(self.cfg.model_path, self.cfg.metric, best_metric * 100))

            self.scheduler.step()
        logging.info('Training finished, best {}: {:.2f} %'.format(self.cfg.metric, best_metric * 100))

    def test_modelnet(self):
        logging.info('Test {} on {}{} ...'.format(self.cfg.model_name, self.cfg.dataset_name, self.cfg.category))
        self.model.load(self.cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Evaluating test set.
        self.metrics.reset()
        with Ctq(self.dataset.test_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Evaluating test set')
                data = data.to(self.device)
                with torch.no_grad():
                    self.model.forward(data)
                self.model.compute_loss()
                tq_loader.set_postfix(internal_loss=self.model.internal_loss.item(),
                                      external_loss=self.model.external_loss.item(),
                                      total_loss=self.model.loss.item())
                self.metrics.update(self.model.y.cpu().numpy(), self.model.y_hat.max(dim=1)[1].cpu().numpy())
        score_dict, _ = self.metrics.get_scores()
        logging.info(
            'Test OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100, score_dict['Mean IoU'] * 100))

    def test_s3dis(self, num_votes=100):
        logging.info('Evaluating {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset_name))
        os.makedirs(self.cfg.save_path) if not os.path.exists(self.cfg.save_path) else None
        test_smooth = 0.95
        self.test_probs = [np.zeros(shape=(t.data.shape[0], cfg.num_classes), dtype=np.float32) for t in
                           self.dataset.test_set.input_trees]
        # statistic label proportions in test set
        class_proportions = np.zeros(self.cfg.num_classes, dtype=np.float32)
        for i, label in enumerate(self.dataset.test_set.label_values):
            class_proportions[i] = np.sum([np.sum(labels == label) for labels in self.dataset.test_set.val_labels])

        # load model checkpoints
        self.model.load(self.cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

        epoch = 0
        last_min = -0.5
        while last_min < num_votes:

            # test one epoch
            with Ctq(self.dataset.test_loader) as tq_loader:
                for i, data in enumerate(tq_loader):
                    tq_loader.set_description('Evaluation')
                    # model inference
                    data = data.to(self.device)
                    with torch.no_grad():
                        logits = self.model(data)                     # get pred
                        y_pred = F.softmax(logits, dim=-1)

                    y_pred = y_pred.cpu().numpy()
                    y_target = data.y.cpu().numpy()                   # get target
                    point_idx = data.point_idx.cpu().numpy()          # the point idx
                    cloud_idx = data.cloud_idx.cpu().numpy()          # the cloud idx
                    batch = data.batch.cpu().numpy()                  # batch indices

                    # compute batch accuracy
                    correct = np.sum(np.argmax(y_pred, axis=1) == y_target)
                    acc = correct / float(np.prod(np.shape(y_target)))      # accurate for each test batch
                    tq_loader.set_postfix(ACC=acc)

                    # y_pred = y_pred.reshape(self.cfg.batch_size, -1, self.cfg.num_classes)      # [B, N, C]
                    for b in range(self.cfg.batch_size):        # for each sample in batch
                        idx = batch == b
                        probs = y_pred[idx]         # [N, C]
                        p_idx = point_idx[idx]      # [N]
                        c_idx = cloud_idx[b]        # int
                        self.test_probs[c_idx][p_idx] = test_smooth * self.test_probs[c_idx][p_idx] \
                                                       + (1 - test_smooth) * probs   # running means

            new_min = np.min(self.dataset.test_set.min_possibility)
            print('Epoch {:3d} end, current min possibility = {:.2f}'.format(epoch, new_min))

            if last_min + 1 < new_min:
                # update last_min
                last_min += 1
                # show vote results
                print('Confusion on sub clouds.')
                confusion_list = []
                num_clouds = len(self.dataset.test_set.input_labels)        # test cloud number
                for i in range(num_clouds):
                    probs = self.test_probs[i]
                    preds = self.dataset.test_set.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                    labels = self.dataset.test_set.input_labels[i]
                    confusion_list += [confusion_matrix(labels, preds, self.dataset.test_set.label_values)]

                # re-group confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                # re-scale with the right number of point per class
                C *= np.expand_dims(class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # compute IoU
                IoUs = iou_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                print(s)

                if int(np.ceil(new_min) % 1) == 0:      # ???
                    print('re-project vote #{:d}'.format(int(np.floor(new_min))))
                    proj_prob_list = []

                    for i in range(num_clouds):
                        proj_idx = self.dataset.test_set.val_proj[i]
                        probs = self.test_probs[i][proj_idx, :]
                        proj_prob_list += [probs]

                    # show vote results
                    print('confusion on full cloud')
                    confusion_list = []
                    for i in range(num_clouds):
                        preds = self.dataset.test_set.label_values[np.argmax(proj_prob_list[i], axis=1)].astype(np.uint8)
                        labels = self.dataset.test_set.val_labels[i]
                        acc = np.sum(preds == labels) / len(labels)
                        print(self.dataset.test_set.input_names[i] + ' ACC:' + str(acc))
                        confusion_list += [confusion_matrix(labels, preds, self.dataset.test_set.label_values)]
                        name = self.dataset.test_set.input_names[i] + '.ply'
                        write_ply(os.path.join(self.cfg.save_path, name), [preds, labels], ['preds', 'labels'])

                    # re-group confusions
                    C = np.sum(np.stack(confusion_list), axis=0)
                    IoUs = iou_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)

                    print(s)
                    print('finished.')
                    return
            epoch += 1
            continue
        return

    def test_semantic3d(self, num_votes=100):
        logging.info('Test {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset_name))
        os.makedirs(self.cfg.save_path) if not os.path.exists(self.cfg.save_path) else None
        test_smooth = 0.98
        self.test_probs = [np.zeros(shape=(t.data.shape[0], cfg.num_classes), dtype=np.float32) for t in
                           self.dataset.test_set.input_trees]

        # load model checkpoints
        self.model.load(self.cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

        epoch = 0
        last_min = -0.5
        while last_min < num_votes:
            # test one epoch
            with Ctq(self.dataset.test_loader) as tq_loader:
                for i, data in enumerate(tq_loader):
                    tq_loader.set_description('Evaluation')
                    # model inference
                    data = data.to(self.device)
                    with torch.no_grad():
                        logits = self.model(data)
                        y_pred = F.softmax(logits, dim=-1)      # get pred probs

                    y_pred = y_pred.cpu().numpy()
                    point_idx = data.point_idx.cpu().numpy()          # the point idx
                    cloud_idx = data.cloud_idx.cpu().numpy()          # the cloud idx
                    batch = data.batch.cpu().numpy()

                    # running means for each epoch on Test set
                    for b in range(self.cfg.batch_size):        # for each sample in batch
                        idx = batch == b
                        probs = y_pred[idx]     # [N, C]
                        p_idx = point_idx[idx]  # [N]
                        c_idx = cloud_idx[b]    # int
                        self.test_probs[c_idx][p_idx] = test_smooth * self.test_probs[c_idx][p_idx] \
                                                        + (1 - test_smooth) * probs  # running means

            # after each epoch
            new_min = np.min(self.dataset.test_set.min_possibility)
            print('Epoch {:3d} end, current min possibility = {:.2f}'.format(epoch, new_min))
            if last_min + 4 < new_min:
                print('Test procedure done, saving predicted clouds ...')
                last_min = new_min
                # projection prediction to original point cloud
                for i, file in enumerate(self.dataset.test_set.test_files):
                    proj_idx = self.dataset.test_set.test_proj[i]           # already get the shape
                    probs = self.test_probs[i][proj_idx, :]                 # same shape with proj_idx
                    # [0 ~ 7] + 1 -> [1 ~ 8], because 0 for unlabeled
                    preds = np.argmax(probs, axis=1).astype(np.uint8) + 1   # back projection
                    # saving prediction results
                    cloud_name = file.split('/')[-1]
                    ascii_name = os.path.join(self.cfg.save_path, self.dataset.test_set.ascii_files[cloud_name])
                    np.savetxt(ascii_name, preds, fmt='%d')
                    print('Save {:s} succeed !'.format(ascii_name))
                print('Done.')
                return
            epoch += 1
        return

    def test(self):
        if 'ModelNet' in self.cfg.dataset_name:
            self.test_modelnet()
        elif 'S3DIS' in self.cfg.dataset_name:
            self.test_s3dis()
        elif 'Semantic3D' in self.cfg.dataset_name:
            self.test_semantic3d()
        else:
            raise ValueError('Not supported dataset!')

    def __call__(self, *args, **kwargs):
        self.trainval() if self.cfg.mode.lower() == 'train' else self.test()

    def __repr__(self):
        return 'Trainer {} on {}, mode={}, batch_size={}'.format(self.cfg.model_name,
                                                                 self.cfg.dataset_name,
                                                                 self.cfg.mode,
                                                                 self.cfg.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ModelNet', help='choose dataset')
    parser.add_argument('--device_id', type=int, default=0, help='choice device [0 or 1]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=16, help='set batch size')
    parser.add_argument('--num_kernels', type=int, default=8, help='number of kernels')
    parser.add_argument('--category', type=str, default='40', help='category for ModelNet, [`10` or `40`]')
    parser.add_argument('--test_area', type=int, default=5, help='test area for S3DIS, [1 - 6]')

    FLAGS = parser.parse_args()
    cfg = getattr(configure, FLAGS.dataset + 'Config')(device_id=FLAGS.device_id,
                                                       mode=FLAGS.mode,
                                                       batch_size=FLAGS.batch_size,
                                                       category=FLAGS.category,
                                                       num_kernels=FLAGS.num_kernels,
                                                       test_area=FLAGS.test_area)
    trainer = Trainer(cfg)
    print(trainer)
    print(trainer.model)
    trainer()
