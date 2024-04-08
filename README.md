# DL-based mismatch removal methods and their used technologies




| **Reference**                 | **Network structure** | **Main technologies** | **Activation function used in weight generating** | **Datasets**             |**Published** |
|:---------------------------:|:---------------------:|:---------------------:|:-----------------------: |:---------------------:  |:--------------:|
|[PointNet](#PointNet)|  Linear               |  MLP                  | Softmax                   |                         | 2017|
|[LFGC](#LFGC)        |  Linear               |  MLP,CN               | ReLU+Tanh                  |   YFCC100M,SUN3D        | 2018|
|[DFENet](#DFE)       |  Linear               |  MLP,IRLS             | Softmax                    |   KITTI,Tanks,Temples   | 2018|
|[N3Net](#N3N)        |  Linear               |  MLP,Differential KNN  | ReLU+Tanh                 |   BSD500                      | 2018|
|[NMNet](#NM)         |  Linear               |  MLP,KNN,IN,CN         | Sigmoid                   | NARROW, WIDE,COLMAP,MULTI   |2019|
|[OANet](#OA)         |  Linear               |  MLP,CN,OA,IRLS         | ReLU+Tanh                | YFCC100M,SUN3D            |2019|
|[LMR](#LMR)         |  Linear               |  MLP, Multiple KNNs       | Sigmoid                   | RS,Retina,DAISY, DTU     |2019|
|[NG-RANSAC](#NG)      |  -                    |    Unsupervised          |      -                   |    YFCC100M,SUN3D        |2019|
|[ULCM](#ULCM)      |  -                    |    Unsupervised          |      -                   |    MiddleburyMultiView,KITTI        |2019|
|[ACNE](#ACNE)      |   Linear                   |       MLP, ACN      |       ReLU+Tanh            |          YFCC100M,SUN3D                | 2020    |
|[LMCNet](#LMC)      |    Linear                 |      MLP,KNN,CN,OA        |   Sigmoid                     |  YFCC100M,SUN3D,DETRAC        |2020|                                               
|[CLNet](#CL)      |   Linear                  |        MLP,GCN      |           ReLU+tanh             |                 YFCC100M,SUN3D      | 2021    |
|[T-Net](#T)      |       T                      |       MLP,CA,CN,OA          |        ReLU+tanh    |                     YFCC100M,SUN3D            |  2021   |
|[GLHA](#GLHA)      | Linear                      |MLP,BACN,CA,Guided Loss               |Softmax       |         YFCC100M,SUN3D |    2021|
|[PESA-Net](#PESA)      |    Linear                 |                            |                        |                       |2022  |




|[MSA-Net](#MSA)      |    Linear                 |       MLP,CN,MSA,OA                |          ReLU+tanh                |             YFCC100M,SUN3D   |  2022   |

|[](#)      |                     |              |                        |                                                         |     |




Abbreviation  
MLP:multilayer perceptron  
CN:context normalization  
IRLS:iteratively reweighted least squares  
KNN: k-nearest neighbors  
IN: instance-norm  
OA: Differentiable Pooling layer, Order and Aware Filtering block, and Differentiable Unpooling layer
CA: channel-wise attention  
BACN: Bayesian attentive context normalization  
ACN: Attentive Context Normalization  
GCN:  
MSA:multi-scale attention  


- <a id="PointNet">[PointNet]</a> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,CVPR'2017[[project]](https://stanford.edu/~rqi/pointnet/)
- <a id="LFGC">[LFGC]</a> Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- <a id="DFE">[DFENet]</a> Deep fundamental matrix estimation, ECCV'2018[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf) [[code]](https://github.com/isl-org/DFE)
- <a id="N3N">[N3Net]</a> Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
- <a id="NM"> [NM-Net]</a> NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences, arXiv'2019 [[pdf]](https://arxiv.org/pdf/1904.00320)
- <a id="OA">[OANet]</a> Learning Two-View Correspondences and Geometry Using Order-Aware Network ICCV'2019 [[code]](https://github.com/zjhthu/OANet)
- <a id="LMR">[LMR]</a>  LMR: Learning A Two-class Classifier for Mismatch Removal, TIP'2019 [[pdf]](https://starainj.github.io/Files/TIP2019-LMR.pdf) [[code]](https://github.com/StaRainJ/LMR)
- <a id="NG">[NG-RANSAC]</a> Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses, ICCV'2019 [[pdf](https://arxiv.org/pdf/1905.04132.pdf)] [[code](https://github.com/vislearn/ngransac)] [[project](https://hci.iwr.uni-heidelberg.de/vislearn/research/neural-guided-ransac/)]
- <a id="ULCM">[ULCM]</a> Unsupervised Learning of Consensus Maximization for 3D Vision Problems, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Probst_Unsupervised_Learning_of_Consensus_Maximization_for_3D_Vision_Problems_CVPR_2019_paper.pdf)
- [ACNe] ACNe: Attentive context normalization for robust permutation-equivariant learning, CVPR'2020[[code]](https://github.com/vcg-uvic/acne)
- [LMCNet] Learnable motion coherence for correspondence pruning,CVPR'2021 [[pdf]](https://arxiv.org/abs/2011.14563)[[code]](https://github.com/liuyuan-pal/LMCNet)
- Point2CN: Progressive two-view correspondence learning via information fusion
- TSSN-Net: Two-step Sparse Switchable Normalization for Learning Correspondences with Heavy Outliers
- [CLNet] Progressive correspondence pruning by consensus learning,ICCV'2021 [[pdf]](https://arxiv.org/abs/2101.00591)
- [TNet] T-Net: Effective permutation-equivariant network for two-view correspondence learning,ICCV'2021[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_T-Net_Effective_Permutation-Equivariant_Network_for_Two-View_Correspondence_Learning_ICCV_2021_paper.pdf)
- [GLHA] Cascade Network with Guided Loss and Hybrid Attention for Finding Good Correspondences,AAAI'2021[[pdf]](https://arxiv.org/abs/2102.00411)[[code]](https://github.com/wenbingtao/GLHA)
- [MSANet]MSA-net: Establishing reliable correspondences by multiscale attention network,TIP'2022[[pdf]](https://guobaoxiao.github.io/papers/TIP2022b(1).pdf)
- [PESA-Net] PESA-Net: Permutation-Equivariant Split Attention Network for correspondence learning,IF'2022[[pdf]](https://guobaoxiao.github.io/papers/IF_2021_PESA-Net.pdf)[[code]](https://github.com/guobaoxiao/PESA-Net)
