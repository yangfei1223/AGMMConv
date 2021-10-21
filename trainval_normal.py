# -*- coding:utf-8 -*-
'''
@Time : 2021/6/6 下午10:04
@Author: yangfei
@File : trainval_normal.py
'''
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from utils import runingScoreNormal, write_ply
from configure import ModelNetNormalConfig

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

        self.metrics = runingScoreNormal()

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
                self.metrics.update(self.model.y, self.model.y_hat, data.batch)

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
                self.metrics.update(self.model.y, self.model.y_hat, data.batch)

    def train(self):
        min_err = 1.0
        # track parameters
        self.model.to(self.device)
        for epoch in range(self.cfg.epochs):
            logging.info('Training epoch: {}, learning rate: {}'.format(epoch, self.scheduler.get_last_lr()[0]))

            # training
            self.train_one_epoch(epoch)
            err = self.metrics.get_scores()
            logging.info('Training error: {:.2f}'.format(err))
            # validation
            self.val_one_epoch(epoch)
            err = self.metrics.get_scores()
            logging.info('Test error: {:.2f}'.format(err))

            cur_err = err
            if cur_err <= min_err:
                min_err = cur_err
                self.model.save(self.cfg.model_path)     # save model
                logging.info('Save {} succeed, min error: {:.2f}!'.format(self.cfg.model_path, min_err))

            self.scheduler.step()
        logging.info('Training finished, min error: {:.2f} %'.format(min_err))

    def test(self):
        logging.info('Normal estimation {} on {}{} ...'.format(self.cfg.model_name, self.cfg.dataset_name, self.cfg.category))
        os.makedirs(self.cfg.save_path) if not os.path.exists(self.cfg.save_path) else None
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
                    pred = self.model.forward(data)
                self.model.compute_loss()
                tq_loader.set_postfix(internal_loss=self.model.internal_loss.item(),
                                      external_loss=self.model.external_loss.item(),
                                      total_loss=self.model.loss.item())
                self.metrics.update(self.model.y, self.model.y_hat, data.batch)
                pos = data.pos.cpu().numpy()
                norm = data.norm.cpu().numpy()
                pred = pred.cpu().numpy()
                filename = '{:06d}.ply'.format(i)
                write_ply(os.path.join(self.cfg.save_path, filename), [pos, norm, pred],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'px', 'py', 'pz'])

        err = self.metrics.get_scores()
        logging.info('Normal estimation error: {:.2f}'.format(err))

    def __call__(self, *args, **kwargs):
        self.train() if self.cfg.mode.lower() == 'train' else self.test()

    def __repr__(self):
        return 'Trainer {} on {}, mode={}, batch_size={}'.format(self.cfg.model_name,
                                                                 self.cfg.dataset_name,
                                                                 self.cfg.mode,
                                                                 self.cfg.batch_size)


if __name__ == '__main__':
    cfg = ModelNetNormalConfig(device_id=0,
                               mode='test',
                               batch_size=1,
                               category='40',
                               num_kernels=8,
                               task='object')
    trainer = Trainer(cfg)
    print(trainer)
    print(trainer.model)
    trainer()
