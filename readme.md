# AMPLE - Implementation
## Vulnerability Detection with Graph Simplification and Enhanced Graph Representation Learning

## Introduction
Prior studies have demonstrated the effectiveness of Deep Learning (DL) in automated software vulnerability detection. Graph Neural Networks (GNNs) have proven effective in learning the graph representations of source code and are commonly adopted by existing DL-based vulnerability detection methods. However, the existing methods are still limited by the fact that GNNs are essentially difficult to handle the connections between long-distance nodes in a code structure graph. Besides, they do not well exploit the multiple types of edges in a code structure graph (such as edges representing data flow and control flow). Consequently, despite achieving state-of-the-art performance, the existing GNN-based methods tend to fail to capture global information (\ie, long-range dependencies among nodes) of code graphs. 

To mitigate these issues, in this paper, we propose a novel vulnerability detection framework with grAph siMplification and enhanced graph rePresentation LEarning, named AMPLE. AMPLE mainly contains two parts: 1) graph simplification, which aims at reducing the distances between nodes by shrinking the node sizes of code structure graphs; 2) enhanced graph representation learning, which involves one edge-aware graph convolutional network module for fusing heterogeneous edge information into node representations and one kernel-scaled representation moule for well capturing the relations between distant graph nodes. Experiments on three public benchmark datasets show that AMPLE outperforms the state-of-the-art methods by 0.39%-35.32% and 7.64%-199.81% with respect to the accuracy and F1 score metrics, respectively. The results demonstrate the effectiveness of AMPLE in learning global information of code graphs for vulnerability detection.

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

**Default settings in AMPLE**:
* Training configs: 
    * batch_size = 64, lr = 0.0001, epoch = 100, patience = 20
    * opt ='RAdam', weight_decay=1e-6

## Preprocessing
We use Joern to generate the code structure graph. It should be noted that the AST and graphs generated by different versions of Joern may have significant differences. So if using the newer versions of Joern to generate code structure graph, the model may have a different performance compared with the results we reported in the paper.

## Folder structure
├── graph_transformer_layers.py
├── mlp_readout.py
├── model.py

## References
[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

