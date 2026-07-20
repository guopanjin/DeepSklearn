<p align="center">
 <img src="assets/deepsklearn-banner.png">
</p>
<h1 align="center">DeepSklearn</h1>
<h3 align="center">All You Need Is DeepSklearn.</h3>
<p align="center">An industrial-grade PyTorch platform for modern deep learning algorithms</p>

## What is DeepSklearn?
DeepSklearn is an industrial-grade PyTorch platform that makes modern deep learning algorithms easy to build, train, evaluate, and deploy in a clean, sklearn-style workflow.

It combines a modular deep learning framework with an interactive Streamlit training dashboard, allowing users to configure datasets, select models, tune hyperparameters, start training, and monitor logs directly from a UI. With DeepSklearn, even beginners can launch deep learning training jobs without writing complex boilerplate training code.

The project is designed to support a growing algorithm zoo, including recommendation models, recall models, multi-task learning models, sequence models, and general neural network architectures. DeepSklearn emphasizes practical engineering, clean abstractions, reusable components, and production-inspired training pipelines.
## training dashboard
![deepsklearn demo](assets/deepsklearn.gif)

## Models
Here is the collection of deep learning and recommendation algorithms supported by this project, categorized into five core domains: generative, multi-task learning, non-sequence, recall, and sequence models.

### 1. Non-Sequence Models
| Model           | Full Name                                                                 | Paper                                                                                                                        | Status |
| :-------------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- | :----: |
| **LR**          | Logistic Regression                                                       | [Cox, 1958] *The Regression Analysis of Binary Sequences*                                                                    |    ✅   |
| **FM**          | Factorization Machines                                                    | [Rendle, 2010] *Factorization Machines*                                                                                      |    ✅   |
| **DNN**         | Deep Neural Network                                                       | -                                                                                                                            |    ✅   |
| **Wide & Deep** | Wide & Deep Learning                                                      | [Cheng et al., 2016] *Wide & Deep Learning for Recommender Systems*                                                          |    ✅   |
| **DeepFM**      | Deep Factorization Machine                                                | [Guo et al., 2017] *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*                                 |    ✅   |
| **DCN**         | Deep & Cross Network                                                      | [Wang et al., 2017] *Deep & Cross Network for Ad Click Predictions*                                                          |    ✅   |
| **DCN-V2**      | Deep & Cross Network V2                                                   | [Wang et al., 2021] *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*     |    ✅   |
| **PNN**         | Product-based Neural Network                                              | [Qu et al., 2016] *Product-based Neural Networks for User Response Prediction*                                               |    ✅   |
| **AutoInt**     | Automatic Feature Interaction Learning via Self-Attentive Neural Networks | [Song et al., 2019] *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*                     |    ✅   |
| **AFM**         | Attentional Factorization Machine                                         | [Xiao et al., 2017] *Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks* |    ✅   |
| **NFM**         | Neural Factorization Machine                                              | [He & Chua, 2017] *Neural Factorization Machines for Sparse Predictive Analytics*                                            |    ✅   |
| **DLRM**        | Deep Learning Recommendation Model                                        | [Naumov et al., 2019] *Deep Learning Recommendation Model for Personalization and Recommendation Systems*                    |    ✅   |
---
### 2. Sequence (User Behavior Sequence Models)
| Model               | Full Name                             | Paper                                                                                                                            | Status |
| :------------------ | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **Average Pooling** | Average Pooling for Sequence Modeling | Common sequence aggregation baseline; no single canonical recommendation paper                                                   |    ✅   |
| **BST**             | Behavior Sequence Transformer         | [Chen et al., 2019] [*Behavior Sequence Transformer for E-commerce Recommendation in Alibaba*](https://arxiv.org/abs/1905.06874) |    ✅   |
| **DIN**             | Deep Interest Network                 | [Zhou et al., 2018] [*Deep Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1706.06978)                |    ✅   |
### 3. Multi-Task Learning
| Model             | Full Name                         | Paper                                                                                                                                                                                       | Status |
| :---------------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----: |
| **Shared Bottom** | Shared-Bottom Multi-Task Learning | [Caruana, 1997] [*Multitask Learning*](https://doi.org/10.1023/A:1007379606734) ([DOI][1])                                                                                                  |    ✅   |
| **MMoE**          | Multi-gate Mixture-of-Experts     | [Ma et al., 2018] [*Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*](https://dl.acm.org/doi/10.1145/3219819.3220007) ([ACM Digital Library][2])      |    ✅   |
| **ESMM**          | Entire Space Multi-Task Model     | [Ma et al., 2018] [*Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate*](https://arxiv.org/abs/1804.07931) ([arXiv][3])                         |    ✅   |
| **PLE**           | Progressive Layered Extraction    | [Tang et al., 2020] [*Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations*](https://doi.org/10.1145/3383313.3412236) ([Papers with Code][4]) |    ✅   |
---

### 4.  Generative models
| Model        | Full Name                                                                  | Paper                                                                                                                                                                            | Status |
| :----------- | :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **GRU4Rec**  | Gated Recurrent Unit for Recommendation                                    | [Hidasi et al., 2016] [*Session-based Recommendations with Recurrent Neural Networks*](https://arxiv.org/abs/1511.06939) ([arXiv][1])                                            |    ✅   |
| **SASRec**   | Self-Attentive Sequential Recommendation                                   | [Kang & McAuley, 2018] [*Self-Attentive Sequential Recommendation*](https://arxiv.org/abs/1808.09781) ([arXiv][2])                                                               |    ✅   |
| **BERT4Rec** | Bidirectional Encoder Representations from Transformers for Recommendation | [Sun et al., 2019] [*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*](https://arxiv.org/abs/1904.06690) ([arXiv][3])            |    ✅   |
| **TIGER**    | Transformer Index for Generative Recommenders                              | [Rajput et al., 2023] [*Recommender Systems with Generative Retrieval*](https://arxiv.org/abs/2305.05065) ([arXiv][4])                                                           |    ✅   |
| **HSTU**     | Hierarchical Sequential Transduction Units                                 | [Zhai et al., 2024] [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) ([arXiv][5]) |    ✅   |
---

### 5. Recall
| Model          | Full Name                                         | Paper                                                                                                                                                                                                                      | Status |
| :------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **Item2Vec**   | Neural Item Embedding for Collaborative Filtering | [Barkan & Koenigstein, 2016] [*Item2Vec: Neural Item Embedding for Collaborative Filtering*](https://arxiv.org/abs/1603.04259)                                                                                             |    ✅   |
| **ItemCF**     | Item-Based Collaborative Filtering                | [Sarwar et al., 2001] [*Item-Based Collaborative Filtering Recommendation Algorithms*](https://doi.org/10.1145/371920.372071)                                                                                              |    ✅   |
| **Swing**      | Swing Item-to-Item Collaborative Filtering        | Alibaba, [*Swing Recommendation Algorithm*](https://www.alibabacloud.com/help/en/airec/what-is-pai-rec/user-guide/swing-algorithm-tools)                                                                                   |    ✅   |
| **Two-Tower**  | Two-Tower Neural Network                          | [Yang et al., 2020] [*Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations*](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/) |    ✅   |
| **YouTubeDNN** | Deep Neural Network for YouTube Recommendations   | [Covington et al., 2016] [*Deep Neural Networks for YouTube Recommendations*](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)                                                              |    ✅   |
---

## Contact
For questions, suggestions, or collaboration, please contact:
Email: guopan.jin@outlook.com
## Citation
If you find DeepSklearn useful in your research or projects, please consider citing it:
```bibtex
@software{jin2026deepsklearn,
  author = {Jin, Guopan},
  title  = {{DeepSklearn}},
  year   = {2026},
  url    = {https://github.com/guopanjin/deepsklearn}
}
```

