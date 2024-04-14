# DL-based mismatch removal methods 

| **Reference**               |  **Main technologies**| **Datasets**             |**Published** |
|:---------------------------:|:---------------------: |:---------------------:  |:--------------:|
|[PointNet](#PointNet)|  MLP                   |                         | 2017|
|[LFGC](#LFGC)        |  MLP,CN                |   YFCC100M,SUN3D        | 2018|
|[DFENet](#DFE)       |  MLP,IRLS              |   KITTI,Tanks,Temples   | 2018|
|[N3Net](#N3N)        | MLP,Differential KNN   |   BSD500                | 2018|
|[NMNet](#NM)         |MLP,KNN,CN          | NARROW, WIDE,COLMAP,MULTI   |2019|
|[OANet](#OA)         | MLP,CN,OA,IRLS         | YFCC100M,SUN3D            |2019|
|[LMR](#LMR)         | MLP, Multiple KNNs       | RS,Retina,DAISY, DTU     |2019|
|[NG-RANSAC](#NG)    |  Unsupervised          |    YFCC100M,SUN3D        |2019|
|[ULCM](#ULCM)      |  Unsupervised          |    Temple,Dino,KITTI        |2019|
|[Superglue](#super)|   Graph Attention,MLP,Sinkhorn |   ScanNet,PhotoTourism,HPatches,Aachen Day-Night   |   2020   |
|[ACNE](#ACNE)      |   MLP, CN,Attention             |    YFCC100M,SUN3D      | 2020    |
|[LMCNet](#LMC)      |   MLP,KNN,CN,OA     |  YFCC100M,SUN3D,DETRAC        |2020| 
|[LSV-ANet](#)      |  KNN,Attention       |  SUIRD,RS,VGG,OXBs  |   2021   |
|[CLNet](#CL)      | MLP,GCN                |  YFCC100M,SUN3D      | 2021    |
|[T-Net](#T)       |  MLP,Attention,CN,OA          |  YFCC100M,SUN3D    |  2021   |
|[GLHA](#GLHA)      |MLP,CN,Attention,Guided Loss    |         YFCC100M,SUN3D |    2021|
|[CAT](#)          |   MLP,CN,Transformer           |        YFCC100M,SUN3D       |      2022 |
|[CSR-Net](#)      |    MLP,CN,Attention         |       MIM,RS,SUIR,VGG,OxBs,MTI        |    2022         |
|[PESA-Net](#PESA)    |  MLP,CN,Attention ,OA   |        YFCC100M,SUN3D     |2022  |
|[MSA-Net](#MSA)      |  MLP,CN,Attention,OA      |             YFCC100M,SUN3D   |  2022   |
|[GANet](#GA)      |   MLP,CN,Graph Attention,OA        |   YFCC100M,SUN3D        |  2022   |
|[MS2DGNet](#MS2DG) |  MLP,CN,Graph Attention     |   YFCC100M,SUN3D            |  2022   |
|[NeFSAC](#NeF)      | MLP,MaxPool          |    KITTI,PhotoTourism   |   2022  |
|[PGFNet](#PGF)      |  MLP,CN,SA,Attention,OA      |   YFCC100M,SUN3D        |  2023   |
|[ANANet](#ANA)      |  MLP, Attention       |     YFCC100M,SUN3D   | 2023    |
|[ConvMatch](#)      |       motion field,CN    |        YFCC100M,SUN3D       |        2023     |
|[ParaFormer](#)      |        MLP, Transformer,Sinkhorn         |      R1M,MegaDepth,YFCC100M,HPatches         |        2023     |
|[IMP](#)      |           MLP,transformer        |        YFCC100M,Scannet,Aachen Day-Night       |        2023     |
|[NCMNet ](#)  |      MLP,CN,OA,GCN             |          YFCC100M,SUN3D     |        2023       |
|[jranet 	](#)      |   MLP，CN，Attention                |      YFCC100M,SUN3D         |     2023          |
|[gcanet](#)      |        MLP,CN,Attention,OA           |        YFCC100M,SUN3D       |      2023         |
|[ARS-MAGSAC](#)    |   MLP,CN,MAGSAC++         |       PhotoTourism，KITTI        |        2023       |
|[SAM](#)      |     MLP,Transformer	         |      Oxford100k,MegaDepth,YFCC100M,HPatches         |       2023        |
|[Htmatch](#)      |     Transformer,GNN,Sinkhorn   |            GL3D,FM-Bench,YFCC100M,ScanNet,Aachen Day-Night    |      2023         |
|[Sganet](#)      |   MLP,CN,Transformer                |      YFCC100M,SUN3D        |       2023        |
|[Emergent corr ](#)      |     Diffusion              |               |        2023       |
|[Amatformer](#)      |     Transformer      |  GL3D，Scannet，FM-Bench，YFCC100M             |     2023          |
|[Roma](#)      |         Diffusion，Transformer           |        MegaDepth，ScanNet，WxBS ，InLoc       |        2024     |
|[Vsformer](#)      |   Transformer,MLP,KNN                |         YFCC100M,SUN3D       |        2024        |
|[resmatch](#)      |           MLP,KNN,GNN,Transformer        |       GL3D,FM-Bench,YFCC100M,ScanNet,Aachen Day-Night        |      2024          |
|[GCT-Net](#)      |          Transformer，GNN         |        YFCC100M,SUN3D       |       2024         |
|[BCLNet](#)      |           OA,Transformer        |       YFCC100M,SUN3D        |        2024        |
|[MSGA-Net](#)      |     Transformer              |         YFCC100M,SUN3D      |        2024        |
|[GIM](#)      |           self-training        |               |        2024        |
|[SSL-Net](#)      |        MLP,CN,Attention,GCN           |      YFCC100M,SUN3D         |      2024          |

Abbreviation  
MLP:multilayer perceptron  
CN:context normalization  
IRLS:iteratively reweighted least squares  
KNN: k-nearest neighbors  
OA: Differentiable Pooling layer, Order and Aware Filtering block, and Differentiable Unpooling layer



GCN: Graph Convolution Network

GA:Graph Attention 
SA:spatial attention


- <a id="PointNet">[PointNet]</a> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,CVPR'2017[[project]](https://stanford.edu/~rqi/pointnet/)
- <a id="LFGC">[LFGC]</a> Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- <a id="DFE">[DFENet]</a> Deep fundamental matrix estimation, ECCV'2018[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf) [[code]](https://github.com/isl-org/DFE)
- <a id="N3N">[N3Net]</a> Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
- <a id="NM"> [NM-Net]</a> NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences, arXiv'2019 [[pdf]](https://arxiv.org/pdf/1904.00320)
- <a id="OA">[OANet]</a> Learning Two-View Correspondences and Geometry Using Order-Aware Network ICCV'2019 [[code]](https://github.com/zjhthu/OANet)
- <a id="LMR">[LMR]</a>  LMR: Learning A Two-class Classifier for Mismatch Removal, TIP'2019 [[pdf]](https://starainj.github.io/Files/TIP2019-LMR.pdf) [[code]](https://github.com/StaRainJ/LMR)
- <a id="NG">[NG-RANSAC]</a> Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses, ICCV'2019 [[pdf](https://arxiv.org/pdf/1905.04132.pdf)] [[code](https://github.com/vislearn/ngransac)] [[project](https://hci.iwr.uni-heidelberg.de/vislearn/research/neural-guided-ransac/)]
- <a id="ULCM">[ULCM]</a> Unsupervised Learning of Consensus Maximization for 3D Vision Problems, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Probst_Unsupervised_Learning_of_Consensus_Maximization_for_3D_Vision_Problems_CVPR_2019_paper.pdf)
- superglue
- [ACNe] ACNe: Attentive context normalization for robust permutation-equivariant learning, CVPR'2020[[code]](https://github.com/vcg-uvic/acne)
- [LMCNet] Learnable motion coherence for correspondence pruning,CVPR'2021 [[pdf]](https://arxiv.org/abs/2011.14563)[[code]](https://github.com/liuyuan-pal/LMCNet)
- LSV-ANet
- [CLNet] Progressive correspondence pruning by consensus learning,ICCV'2021 [[pdf]](https://arxiv.org/abs/2101.00591)
- [TNet] T-Net: Effective permutation-equivariant network for two-view correspondence learning,ICCV'2021[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_T-Net_Effective_Permutation-Equivariant_Network_for_Two-View_Correspondence_Learning_ICCV_2021_paper.pdf)
- [GLHA] Cascade Network with Guided Loss and Hybrid Attention for Finding Good Correspondences,AAAI'2021[[pdf]](https://arxiv.org/abs/2102.00411)[[code]](https://github.com/wenbingtao/GLHA)
- [MSANet]MSA-net: Establishing reliable correspondences by multiscale attention network,TIP'2022[[pdf]](https://guobaoxiao.github.io/papers/TIP2022b(1).pdf)
- cat
- CSR-Net
- [PESA-Net] PESA-Net: Permutation-Equivariant Split Attention Network for correspondence learning,IF'2022[[pdf]](https://guobaoxiao.github.io/papers/IF_2021_PESA-Net.pdf)[[code]](https://github.com/guobaoxiao/PESA-Net)
- [GANet] Learning for mismatch removal via graph attention networks,ISPRS J PHOTOGRAMM'2022,[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/361865594_Learning_for_mismatch_removal_via_graph_attention_networks/links/62ce43a06151ad090b9794dd/Learning-for-mismatch-removal-via-graph-attention-networks.pdf)[[code]](https://github.com/StaRainJ/Code-of-GANet)
- [MS2DGNet] MS2DGNet: Progressive correspondence learning via multiple sparse semantics dynamic graph,CVPR'2022[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Dai_MS2DG-Net_Progressive_Correspondence_Learning_via_Multiple_Sparse_Semantics_Dynamic_Graph_CVPR_2022_paper.pdf)[[code]](https://github.com/changcaiyang/MS2DG-Net)
- [NeFSAC] NeFSAC: Neurally filtered minimal samples,ECCV'2022[[pdf]](https://arxiv.org/abs/2207.07872)[[code]](https://github.com/cavalli1234/NeFSAC)
- [PGFNet]]PGFNet: Preference-guided filtering network for two-view correspondence learning,TIP'2023[[pdf]](https://ieeexplore.ieee.org/document/10041834)[[code]](https://github.com/guobaoxiao/PGFNet)
- [ANA-Net]Learning second-order attentive context for efficient correspondence pruning,AAAI'2023[[pdf]](https://arxiv.org/abs/2303.15761)[[code]](https://github.com/DIVE128/ANANet)
- convmatch
- paraformer
- imp
- NCMNet 
- jranet
- gcanet  github
- ARS-MAGSAC
- SAM
- Htmatch
- Sganet
- Emergent corr
- Amatformer
- Roma
- Vsformer
- resmatch
- GCT-Net
- BCLNet
- MSGA-Net
- GIM
- SSL-Net
