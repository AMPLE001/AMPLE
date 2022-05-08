# DETLA - Implementation
## Software Vulnerability Detection with Graph Simplification and a Novel Graph Neural Network

## Introduction
Prior studies have demonstrated the usefulness of deep learning (DL) and structural source code information in software vulnerability detection. They generally adopt graph neural networks (GNNs) for learning the graph representations of code. However, it has been found that GNNs are sensitive to local information in graph representation learning, and are difficult to capture the knowledge of graphs with large numbers of nodes. These issues could hamper the performance of GNNs in software vulnerability detection. To mitigate these issues, in this paper we propose a novel vulnerability DEtection approach with graph simipLificaTion and scAled GNNs, named DELTA. DELTA contains three major modules: 1) Graph Simplification module, which aims at shrinking the node sizes of code structure graphs while ensuring the code semantics; 2) Edge-Aware Graph Convolutional Network module, which incorporates the edge types for enhancing the local node representations; and 3) Kernel-Scaled Code Representation module, which scales up the  convolution kernel size to simultaneously focus on the global and
local information. Experiments on three benchmark datasets show  that DELTA outperforms the baselines by 0.04%-31.57% and 7.64%-195.18% with respect to the accuracy and F1 score, respectively.

## Full code
Specific codes will be released publicly when the paper gets accepted. Here we give Graph Simplification module, Edge-Aware Graph Convolutional Network module and Kernel-Scaled Code Representation module code.

## Dataset
To investigate the effectiveness of DELTA, we adopt three vulnerability datasets from these paper: 
* Fan et al. [1]: <https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing>
* Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy
* FFMPeg+Qemu [3]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

## Requirement
Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:
* torch  (==1.9.0)
* dgl  (==0.7.2)
* numpy  (==1.22.3)
* sklearn  (==0.0)
* pandas  (==1.4.1)
* tqdm

**Default settings in DETLA**:
* Training configs: 
    * batch_size = 64, lr = 0.0001, epoch = 100, patience = 20
    * opt ='RAdam', weight_decay=1e-6

## Folder structure
├── graph_transformer_layers.py
├── mlp_readout.py
├── model.py

## References
[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

