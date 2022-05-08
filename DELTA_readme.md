# DETLA - Implementation
## Software Vulnerability Detection with Graph Simplification and aNovel Graph Neural Network

## Introduction
Prior studies have demonstrated the usefulness of deep learning (DL) and structural source code information in software vulnerability detection. They generally adopt graph neural networks (GNNs) for learning the graph representations of code. However, it has been found that GNNs are sensitive to local information in graph representation learning, and are difficult to capture the knowledge of graphs with large numbers of nodes. These issues could hamper the performance of GNNs in software vulnerability detection. To mitigate these issues, in this paper we propose a novel vulnerability DEtection approach with graph simipLificaTion and scAled GNNs, named DELTA. DELTA contains three major modules: 1) Graph Simplification module, which aims at shrinking the node sizes of code structure graphs while ensuring the code semantics; 2) Edge-Aware Graph Convolutional Network module, which incorporates the edge types for enhancing the local node representations; and 3) Kernel-Scaled Code Representation module, which scales up the  convolution kernel size to simultaneously focus on the global and
local information. Experiments on three benchmark datasets show  that DELTA outperforms the baselines by 0.04%-31.57% and 7.64%-195.18% with respect to the accuracy and F1 score, respectively.

## Full code
Specific codes will be released publicly when the paper gets accepted. Here we give Graph Simplification module, Edge-Aware Graph Convolutional Network module and Kernel-Scaled Code Representation module code.

## Dataset
To investigate the effectiveness of DELTA, we adopt the three vulnerability datasets, including FFMPeg+Qemu, Reveal and Fan et al.

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



