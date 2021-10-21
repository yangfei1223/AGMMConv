import os, glob
import numpy as np
from utils import runningScore, read_ply, write_ply

root = 'RUNS/results/S3DISDataset-CRF'
metrics = runningScore(13, ignore_index=-1)

for area in range(5, 6):
    print('Collecting Area {} ...'.format(area))
    path = os.path.join(root, 'Area_{}'.format(area))
    for filename in os.listdir(path):
        print('Processing {}.'.format(filename))
        data = read_ply(os.path.join(path, filename))
        preds, labels = data['preds'], data['labels']
        metrics.update(labels, preds)

score_dict, cls_iou = metrics.get_scores()
print('OA: {:.2f} %, mACC: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100,
                                                            score_dict['Mean Acc'] * 100,
                                                            score_dict['Mean IoU'] * 100))
print('Class IoU:')
print(cls_iou)






