# DL-based mismatch removal methods and their used technologies




| **Reference**       | **Network structure** | **Main technologies** | **Activation function** | **Published** |
|:-------------------:|:---------------------:|:---------------------:|:-----------------------:|:--------------:|
|[PointNet](#PointNet)|  Linear               |  MLP                  | Softmax                  | 2017|
|[LFGC](#LFGC)        |  Linear               |  MLP,CN               | ReLU+Tanh                | 2018|
|[DFENet](#DFE)       |  Linear               |  MLP,IRLS             | Softmax                  | 2018|
|[N3Net](#N3N)        |  Linear               |  MLP,Differential KNN  | ReLU+Tanh                | 2018|




Abbreviation  
MLP:multilayer perceptron  
CN:context normalization
IRLS:iteratively reweighted least squares
KNN: k-nearest neighbors


- <a id="PointNet">[PointNet]</a> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,CVPR'2017[[project]](https://stanford.edu/~rqi/pointnet/)
- <a id="LFGC">[LFGC]</a> Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- <a id="DFE">[DFENet]</a> Deep fundamental matrix estimation, ECCV'2018[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf) [[code]](https://github.com/isl-org/DFE)
- <a id="N3N">[N3Net]</a>Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
