# -*- coding:utf-8 -*-
'''
@Time : 2020/9/13 下午4:26
@Author: yangfei
@File : metrics.py
'''

# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch
import torch.nn.functional as F


def iou_from_confusions(confusions):
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TPFN = np.sum(confusions, axis=-1)
    TPFP = np.sum(confusions, axis=-2)

    IoU = TP / (TPFP + TPFN - TP + 1e-6)

    mask = TPFN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    IoU += mask * mIoU

    return IoU


class runningScore(object):
    def __init__(self, n_classes, ignore_index=-1):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) & (label_true != self.ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        if label_trues.ndim == 1:
            self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)
        else:
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class runingScoreNormal(object):
    def __init__(self):
        self.total_err = 0
        self.counts = 0

    def update(self, label_trues, label_preds, batch_idx):
        cos_dist = 1. - F.cosine_similarity(label_trues, label_preds, dim=-1).abs()
        batches = torch.unique(batch_idx)
        for batch in batches:
            mask = batch_idx == batch
            err = cos_dist[mask].mean()
            self.total_err += err
        self.counts += len(batches)

    def get_scores(self):
        return self.total_err / self.counts

    def reset(self):
        self.total_err = 0
        self.counts = 0


if __name__ == '__main__':
    score = runningScore()
    print(score)
