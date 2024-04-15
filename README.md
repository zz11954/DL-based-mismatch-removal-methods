# DL-based mismatch removal methods 

| **Reference**               |  **Main technologies**| **Datasets**             |**Published** |
|:---------------------------:|:---------------------: |:---------------------:  |:--------------:|
|[PointNet](#PointNet)|  MLP                   |                         | 2017|
|[LFGC](#LFGC)        |  MLP,CN                |   YFCC100M,SUN3D        | 2018|
|[DFENet](#DFE)       |  MLP,IRLS              |   KITTI,Tanks,Temples   | 2018|
|[N3Net](#N3N)        | MLP,Differential KNN   |   BSD500                | 2018|
|[NMNet](#NM)         |MLP,KNN,CN              | NARROW, WIDE,COLMAP,MULTI   |2019|
|[OANet](#OA)         | MLP,CN,OA,IRLS         | YFCC100M,SUN3D            |2019|
|[LMR](#LMR)         | MLP, Multiple KNNs       | RS,Retina,DAISY, DTU     |2019|
|[NG-RANSAC](#NG)    |  Unsupervised          |    YFCC100M,SUN3D        |2019|
|[ULCM](#ULCM)      |  Unsupervised          |    Temple,Dino,KITTI        |2019|
|[SuperGlue](#super)|   Graph Attention,MLP,Sinkhorn |   ScanNet,PhotoTourism,HPatches,Aachen Day-Night   |   2020   |
|[ACNE](#ACNE)      |   MLP, CN,Attention             |    YFCC100M,SUN3D      | 2020    |
|[LMCNet](#LMC)      |   MLP,KNN,CN,OA     |  YFCC100M,SUN3D,DETRAC        |2020| 
|[LSV-ANet](#LSVA)      |  KNN,Attention       |  SUIRD,RS,VGG,OXBs  |   2021   |
|[CLNet](#CL)      | MLP,GCN                |  YFCC100M,SUN3D      | 2021    |
|[T-Net](#T)       |  MLP,Attention,CN,OA          |  YFCC100M,SUN3D    |  2021   |
|[GLHA](#GLHA)      |MLP,CN,Attention,Guided Loss    |         YFCC100M,SUN3D |    2021|
|[CAT](#CAT)          |   MLP,CN,Transformer           |        YFCC100M,SUN3D       |      2022 |
|[CSR-Net](#CSR)      |    MLP,CN,Attention         |       MIM,RS,SUIR,VGG,OxBs,MTI        |    2022         |
|[PESA-Net](#PESA)    |  MLP,CN,Attention ,OA   |        YFCC100M,SUN3D     |2022  |
|[MSA-Net](#MSA)      |  MLP,CN,Attention,OA      |             YFCC100M,SUN3D   |  2022   |
|[GANet](#GANet)      |   MLP,CN,Graph Attention,OA        |   YFCC100M,SUN3D        |  2022   |
|[MS2DGNet](#MS2DGNet) |  MLP,CN,Graph Attention     |   YFCC100M,SUN3D            |  2022   |
|[NeFSAC](#NeFSAC)      | MLP,MaxPool          |    KITTI,PhotoTourism   |   2022  |
|[PGFNet](#PGFNet)      |  MLP,CN,SA,Attention,OA      |   YFCC100M,SUN3D        |  2023   |
|[ANANet](#ANA)      |  MLP, Attention       |     YFCC100M,SUN3D   | 2023    |
|[ConvMatch](#conv)      |       motion field,CN    |        YFCC100M,SUN3D       |        2023     |
|[ParaFormer](#ParaFormer)      |        MLP, Transformer,Sinkhorn         |      R1M,MegaDepth,YFCC100M,HPatches         |        2023     |
|[IMP](#IMP)      |           MLP,transformer        |        YFCC100M,Scannet,Aachen Day-Night       |        2023     |
|[NCMNet](#NCMNet)  |      MLP,CN,OA,GCN             |          YFCC100M,SUN3D     |        2023       |
|[JRANet ](#JRANet)      |   MLP,CN,Attention                |      YFCC100M,SUN3D         |     2023          |
|[GCANet](#GCANet)      |        MLP,CN,Attention,OA           |        YFCC100M,SUN3D       |      2023         |
|[ARS-MAGSAC](#ARS)    |   MLP,CN,MAGSAC++         |       PhotoTourism，KITTI        |        2023       |
|[SAM](#SAM)      |     MLP,Transformer	         |      Oxford100k,MegaDepth,YFCC100M,HPatches         |       2023        |
|[HTMatch](#HTMatch)      |     Transformer,GNN,Sinkhorn   |            GL3D,FM-Bench,YFCC100M,ScanNet,Aachen Day-Night    |      2023         |
|[SGA-Net](#SGA)      |   MLP,CN,Transformer                |      YFCC100M,SUN3D        |       2023        |
|[DIFT ](#DIFT)      |     Diffusion              |               |        2023       |
|[AMatFormer](#AMatFormer)      |     Transformer      |  GL3D，Scannet，FM-Bench，YFCC100M             |     2023          |
|[RoMa](#RoMa)      |         Diffusion,Transformer           |        MegaDepth，ScanNet，WxBS ，InLoc       |        2024     |
|[VSFormer](#VSFormer)      |   Transformer,MLP,KNN                |         YFCC100M,SUN3D       |        2024        |
|[ResMatch](#ResMatch)      |           MLP,KNN,GNN,Transformer        |       GL3D,FM-Bench,YFCC100M,ScanNet,Aachen Day-Night        |      2024          |
|[GCT-Net](#GCT)      |          Transformer,GNN         |        YFCC100M,SUN3D       |       2024         |
|[BCLNet](#BCLNet)      |           OA,Transformer        |       YFCC100M,SUN3D        |        2024        |
|[MSGA-Net](#MSGA)      |     Transformer              |         YFCC100M,SUN3D      |        2024        |
|[GIM](#GIM)      |           self-training        |               |        2024        |
|[SSL-Net](#SSL)      |        MLP,CN,Attention,GCN           |      YFCC100M,SUN3D         |      2024          |

Abbreviation  
MLP:multilayer perceptron  
CN:context normalization  
IRLS:iteratively reweighted least squares  
KNN: k-nearest neighbors  
OA: Differentiable Pooling layer, Order and Aware Filtering block, and Differentiable Unpooling layer  
GCN: Graph Convolution Network  



- <a id="PointNet">[PointNet]</a> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,CVPR'2017[[project]](https://stanford.edu/~rqi/pointnet/)
- <a id="LFGC">[LFGC]</a> Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- <a id="DFE">[DFENet]</a> Deep fundamental matrix estimation, ECCV'2018[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf) [[code]](https://github.com/isl-org/DFE)
- <a id="N3N">[N3Net]</a> Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
- <a id="NM"> [NM-Net]</a> NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences, arXiv'2019 [[pdf]](https://arxiv.org/pdf/1904.00320)
- <a id="OA">[OANet]</a> Learning Two-View Correspondences and Geometry Using Order-Aware Network,ICCV'2019 [[code]](https://github.com/zjhthu/OANet)
- <a id="LMR">[LMR]</a>  LMR: Learning A Two-class Classifier for Mismatch Removal, TIP'2019 [[pdf]](https://starainj.github.io/Files/TIP2019-LMR.pdf) [[code]](https://github.com/StaRainJ/LMR)
- <a id="NG">[NG-RANSAC]</a> Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses, ICCV'2019 [[pdf](https://arxiv.org/pdf/1905.04132.pdf)] [[code](https://github.com/vislearn/ngransac)] [[project](https://hci.iwr.uni-heidelberg.de/vislearn/research/neural-guided-ransac/)]
- <a id="ULCM">[ULCM]</a> Unsupervised Learning of Consensus Maximization for 3D Vision Problems, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Probst_Unsupervised_Learning_of_Consensus_Maximization_for_3D_Vision_Problems_CVPR_2019_paper.pdf)
- <a id="super">[SuperGlue]</a> SuperGlue: Learning Feature Matching with Graph Neural Networks, CVPR'2020 [[code]](https://github.com/magicleap/SuperGluePretrainedNetwork)
- <a id="ACNE"> [ACNe] </a> ACNe: Attentive context normalization for robust permutation-equivariant learning, CVPR'2020[[code]](https://github.com/vcg-uvic/acne)
- <a id="LMC"> [LMCNet]</a> Learnable motion coherence for correspondence pruning,CVPR'2021 [[pdf]](https://arxiv.org/abs/2011.14563)[[code]](https://github.com/liuyuan-pal/LMCNet)
- <a id="LSVA">[LSV-ANet]</a> LSV-ANet: Deep Learning on Local Structure Visualization for Feature Matching,TGRS'2021[[pdf]](https://ieeexplore.ieee.org/document/9377555)
- <a id="CL">[CLNet]</a> Progressive correspondence pruning by consensus learning,ICCV'2021 [[pdf]](https://arxiv.org/abs/2101.00591)
- <a id="T">[TNet]</a> T-Net: Effective permutation-equivariant network for two-view correspondence learning,ICCV'2021[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_T-Net_Effective_Permutation-Equivariant_Network_for_Two-View_Correspondence_Learning_ICCV_2021_paper.pdf)
- <a id="GLHA">[GLHA]</a> Cascade Network with Guided Loss and Hybrid Attention for Finding Good Correspondences,AAAI'2021[[pdf]](https://arxiv.org/abs/2102.00411)[[code]](https://github.com/wenbingtao/GLHA)
- <a id="MSA">[MSANet]</a> MSA-net: Establishing reliable correspondences by multiscale attention network,TIP'2022[[pdf]](https://guobaoxiao.github.io/papers/TIP2022b(1).pdf)
- <a id="CAT">[CAT]</a> Correspondence Attention Transformer: A Context-sensitive Network for Two-view Correspondence Learning, TMM'2022[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/359451839_Correspondence_Attention_Transformer_A_Context-sensitive_Network_for_Two-view_Correspondence_Learning/links/62ce44b3b261d22751eb64d4/Correspondence-Attention-Transformer-A-Context-Sensitive-Network-for-Two-View-Correspondence-Learning.pdf) [[code]](https://github.com/jiayi-ma/CorresAttnTransformer)
- <a id="CSR">[CSR-net]</a>CSR-net: Learning adaptive context structure representation for robust feature correspondence,TIP'2022[[pdf]](https://ieeexplore.ieee.org/document/9758641)
- <a id="PESA">[PESA-Net]</a> PESA-Net: Permutation-Equivariant Split Attention Network for correspondence learning,IF'2022[[pdf]](https://guobaoxiao.github.io/papers/IF_2021_PESA-Net.pdf)[[code]](https://github.com/guobaoxiao/PESA-Net)
- <a id="GANet">[GANet]</a> Learning for mismatch removal via graph attention networks,ISPRS J PHOTOGRAMM'2022,[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/361865594_Learning_for_mismatch_removal_via_graph_attention_networks/links/62ce43a06151ad090b9794dd/Learning-for-mismatch-removal-via-graph-attention-networks.pdf)[[code]](https://github.com/StaRainJ/Code-of-GANet)
- <a id="MS2DGNet">[MS2DGNet]</a>  MS2DGNet: Progressive correspondence learning via multiple sparse semantics dynamic graph,CVPR'2022[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Dai_MS2DG-Net_Progressive_Correspondence_Learning_via_Multiple_Sparse_Semantics_Dynamic_Graph_CVPR_2022_paper.pdf)[[code]](https://github.com/changcaiyang/MS2DG-Net)
- <a id="NeFSAC">[NeFSAC]</a>  NeFSAC: Neurally filtered minimal samples,ECCV'2022[[pdf]](https://arxiv.org/abs/2207.07872)[[code]](https://github.com/cavalli1234/NeFSAC)
- <a id="PGFNet">[PGFNet]]</a> PGFNet: Preference-guided filtering network for two-view correspondence learning,TIP'2023[[pdf]](https://ieeexplore.ieee.org/document/10041834)[[code]](https://github.com/guobaoxiao/PGFNet)
- <a id="ANA">[ANA-Net]</a>Learning second-order attentive context for efficient correspondence pruning,AAAI'2023[[pdf]](https://arxiv.org/abs/2303.15761)[[code]](https://github.com/DIVE128/ANANet)
- <a id="conv">[ConvMatch]</a> ConvMatch: Rethinking Network Design for Two-View Correspondence Learning, AAAI'2023 [[code]](https://github.com/SuhZhang/ConvMatch)
- <a id="ParaFormer">[ParaFormer]</a> ParaFormer: Parallel Attention Transformer for Efficient Feature Matching,AAAI'2023[[pdf]](https://arxiv.org/abs/2303.00941)
- <a id="IMP">[IMP]</a> IMP: Iterative Matching and Pose Estimation with Adaptive Pooling, CVPR'2023 [[code]](https://github.com/feixue94/imp-release)
- <a id="NCMNet">[NCMNet] </a>Progressive Neighbor Consistency Mining for Correspondence Pruning, CVPR'2023 [[code]](https://github.com/xinliu29/NCMNet)
- <a id="JRANet">[JRANet] </a>JRA-net: Joint representation attention network for correspondence learning,PR'2023[[pdf]](https://www.sciencedirect.com/science/article/pii/S0031320322006598)
- <a id="GCANet">[GCANet] </a>Learning for feature matching via graph context attention,TGARS'2023[[pdf]](https://ieeexplore.ieee.org/document/10075633)[[code]](https://github.com/guobaoxiao/GCA-Net)
- <a id="ARS">[ARS-MAGSAC]</a> Adaptive Reordering Sampler with Neurally Guided MAGSAC,ICCV'2023[[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Adaptive_Reordering_Sampler_with_Neurally_Guided_MAGSAC_ICCV_2023_paper.pdf)[[code]](https://github.com/weitong8591/ars_magsac)
- <a id="SAM">[SAM] </a>Scene-Aware Feature Matching,ICCV'2023[[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Scene-Aware_Feature_Matching_ICCV_2023_paper.pdf)
- <a id="HTMatch">[HTMatch] </a>HTMatch: An efficient hybrid transformer based graph neural network for local feature matching,SP'2023[[pdf]](https://www.sciencedirect.com/science/article/pii/S016516842200398X)
- <a id="SGA">[SGA-Net] </a>SGA-Net: A Sparse Graph Attention Network for Two-View Correspondence Learning,TCSVT'2023[[pdf]](https://ieeexplore.ieee.org/abstract/document/10124002)
- <a id="DIFT">[DIFT]</a>Emergent Correspondence from Image Diffusion,NeurIPS'2023[[pdf]](https://arxiv.org/abs/2306.03881)[[code]](https://github.com/Tsingularity/dift)
- <a id="AMatFormer">[AMatFormer]</a> AMatFormer: Efficient Feature Matching via Anchor Matching Transformer,TMM'2023[[pdf]](https://arxiv.org/pdf/2305.19205.pdf)
- <a id="RoMa">[RoMa]</a> RoMa: Robust Dense Feature Matching, CVPR'2024 [[pdf]](https://arxiv.org/abs/2305.15404) [[code]](https://github.com/Parskatt/RoMa)
- <a id="VSFormer">[VSFormer] </a>VSFormer: Visual-Spatial Fusion Transformer for Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/pdf/2312.08774.pdf)[[code]](https://github.com/sugar-fly/VSFormer)
- <a id="ResMatch">[ResMatch] </a>ResMatch: Residual Attention Learning for Local Feature Matching,AAAI'2024[[pdf]](https://arxiv.org/abs/2307.05180)[[code]](https://github.com/ACuOoOoO/ResMatch)
- <a id="GCT"> [GCT-Net] </a>Graph Context Transformation Learning for Progressive Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/abs/2312.15971)
- <a id="BCLNet">[BCLNet] </a>BCLNet: Bilateral Consensus Learning for Two-View Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/abs/2401.03459)
- <a id="MSGA">[MSGA-Net] </a>MSGA-Net: Progressive Feature Matching via Multi-layer Sparse Graph Attention,TCSVT'2024[[pdf]](https://ieeexplore.ieee.org/document/10439184)
- <a id="GIM">[GIM] </a>GIM: Learning Generalizable Image Matcher From Internet Videos,ICLR'2024[[pdf]](https://arxiv.org/abs/2402.11095)[[code]](https://github.com/xuelunshen/gim)
- <a id="SSL">[SSL-Net]</a> SSL-Net: Sparse semantic learning for identifying reliable correspondences,PR'2024[[pdf]](https://www.sciencedirect.com/science/article/pii/S0031320323007367)[[code]](https://github.com/guobaoxiao/SSL-Net)
  
