<p>
<img src="assets/deepsklearn-banner.png">
</p>
<h1 align="center">DeepSklearn</h1>
<h3 align="center">All You Need Is DeepSklearn.</h3>
<p align="center">An industrial-grade PyTorch platform for modern deep learning algorithms</p>

## What is DeepSklearn?
DeepSklearn is an industrial-grade PyTorch platform for modern deep learning algorithms. It provides sklearn-style unified APIs for datasets, features, models, losses, metrics, optimizers, and trainers, enabling researchers and engineers to build, train, evaluate, and extend deep learning models with clean abstractions and reproducible workflows.
DeepSklearn is designed to support a growing algorithm zoo, including recommendation models, multi-task learning models, sequence models, and general neural network architectures. It emphasizes practical engineering, modular design, and production-inspired training pipelines.

## Models

Here is the collection of deep learning and recommendation algorithms supported by this project, categorized into five core domains: generative, multi-task learning, non-sequence, recall, and sequence models.

### 1. Generative Models
| Model | Full Name | Paper | Status |
| :--- | :--- | :--- | :---: |
| **IRGAN** | Information Retrieval GAN | [Wang et al., 2017] [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models](https://arxiv.org/abs/1705.10513) | ✅ |
| **Multi-VAE** | Variational Autoencoders for CF | [Liang et al., 2018] [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) | ✅ |
| **DiffRec** | Diffusion Recommender Model | [Wang et al., 2023] [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971) | ✅ |

---

### 2. Multi-Task Learning
| Model | Full Name | Paper | Status |
| :--- | :--- | :--- | :---: |
| **ESMM** | Entire Space Multi-Task Model | [Ma et al., 2018] [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) | ✅ |
| **MMoE** | Multi-gate Mixture-of-Experts | [Ma et al., 2018] [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) | ✅ |
| **PLE** | Progressive Layered Extraction | [Tang et al., 2020] [Progressive Layered Extraction (PLE): A Improved Multi-task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) | ✅ |

---

### 3. Non-Sequence (Classic CTR Models)
| Model | Full Name | Paper | Status |
| :--- | :--- | :--- | :---: |
| **LR** | Logistic Regression | *Classic Machine Learning* | ✅ |
| **FM** | Factorization Machines | [Rendle, 2010] [Factorization Machines](https://ieeexplore.ieee.org/document/5694074) | ✅ |
| **FFM** | Field-aware Factorization Machines | [Juan et al., 2016] [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/doi/10.1145/2959100.2959134) | ✅ |
| **FNN** | Factorization-supported Neural Network | [Zhang et al., 2016] [Deep Learning over Multi-field Categorical Data](https://arxiv.org/abs/1601.02376) | ✅ |
| **PNN** | Product-based Neural Network | [Qu et al., 2016] [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) | ✅ |
| **Wide & Deep** | Wide & Deep Learning | [Cheng et al., 2016] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | ✅ |
| **DeepFM** | DeepFM | [Guo et al., 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | ✅ |
| **DCN** | Deep & Cross Network | [Wang et al., 2017] [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | ✅ |
| **DCN V2** | Deep & Cross Network V2 | [Wang et al., 2021] [DCN V2: Improved Deep & Cross Network for Feature Crosses in Recommendation Systems](https://arxiv.org/abs/2008.13535) | ✅ |
| **NFM** | Neural Factorization Machines | [He et al., 2017] [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) | ✅ |
| **AFM** | Attentional Factorization Machine | [Xiao et al., 2017] [Attentional Factorization Machines: Learning Neural Network Effects of Features](https://arxiv.org/abs/1708.04617) | ✅ |
| **xDeepFM** | eXtreme Deep Factorization Machine | [Lian et al., 2018] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) | ✅ |
| **AutoInt** | Automatic Feature Interaction | [Song et al., 2019] [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | ✅ |
| **FiBiNET** | Feature Importance Bilinear Interaction Network | [Huang et al., 2019] [FiBiNET: Combining Feature Importance and Bilinear Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) | ✅ |
| **FLEN** | Field-Leveraged Embedding Network | [An et al., 2020] [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690) | ✅ |

---

### 4. Recall
| Model | Full Name | Paper | Status |
| :--- | :--- | :--- | :---: |
| **DSSM** | Deep Structured Semantic Models | [Huang et al., 2013] [Deep Structured Semantic Models for Web Search using Clickthrough Data](https://dl.acm.org/doi/10.1145/2505515.2505665) | ✅ |
| **YoutubeDNN** | Deep Neural Networks for YouTube Recommendations | [Covington et al., 2016] [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) | ✅ |
| **NCF** | Neural Collaborative Filtering | [He et al., 2017] [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | ✅ |
| **MIND** | Multi-Interest Network with Dynamic Routing | [Li et al., 2019] [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.00347) | ✅ |

---

### 5. Sequence (User Behavior Sequence Models)
| Model | Full Name | Paper | Status |
| :--- | :--- | :--- | :---: |
| **DIN** | Deep Interest Network | [Zhou et al., 2018] [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) | ✅ |
| **DIEN** | Deep Interest Evolution Network | [Zhou et al., 2019] [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) | ✅ |
| **DSIN** | Deep Session Interest Network | [Feng et al., 2019] [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) | ✅ |
| **SASRec** | Self-Attentive Sequential Recommendation | [Kang et al., 2018] [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) | ✅ |
| **BST** | Behavior Sequence Transformer | [Chen et al., 2019] [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) | ✅ |