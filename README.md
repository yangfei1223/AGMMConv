# AGMMConv
This repository is the implementation of "Adaptive GMM Convolution for Point Cloud Learning"
## 1. Setup
### 1) Building
```bash
cd utils
sh complile_op.sh
```
### 2) Dependency
This repository is partially dependent on 'pytorch', 'torch_geometric' and 'torch_points3d'.

## 2. Runing
Please see 'trainval.y' for classification and segmentation tasks, and 'trainval_normal.py' for normal estimation task.\
Please see 'configure.py' for the settings. \
And see the 'dataset' directories for the data processing. 

## 3. Acknowledgement
Part of the codes refers to the KPConv and RandLA-Net

## 4. Others
This repository will continue updating. 
