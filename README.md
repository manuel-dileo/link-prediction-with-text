# Link prediction with text
This repository contains the source code used in three different works:
- Dileo, M., Ba, C. T., Zignani, M., and Gaito, S. The impact of textual context on link prediction in online social networks, 2022. The 13th International Conference on Complex Networks, Exeter, United Kingdom, May 30 - June 1
- Dileo, M., Ba, C. T., Zignani, M., and Gaito, S. Link prediction with text in online social networks: the role of textual content on high-resolution temporal data. The 25th International Conference on Discovery Science, Montpellier, France, 10-12 October
- Dileo, M., Ba, C. T., Zignani, M., and Gaito, S. Link prediction in blockchain online social networks with textual information, 2022. The 11th International Conference on Complex Networks and their Applications, Palermo, Italy, 8-10 November

## Description
We investigated the role of text on link formation, an important task as text could improve prediction and give insight on link formation process. To this end, we performed link prediction with text on a temporal attributed network. We relied on Steemit, a blockchain-based online social network, that allows the retrieval of high-resolution temporal information but lacks user attributes due to data control and privacy reasons. We have provided a methodology to use text information alongside traditional structural information and a temporal framework to train and test the models.

First, we showed that the combination of structural and textual features improved prediction performance in terms of F1 score on the traditional supervised models. Then, we showed that some textual features are considered more important than the most important structural features.  This is important as we tested on two time intervals where the network changes a lot, hence a dominance of the structural features in terms of importance could have led to poor performance. 

GNNs reach an AUROC score of 0.97 working naturally on graph-structured data and using textual information as node features. Textual features enhance the performance of a GNN that works without node features while if the features are augmented through structural information, such as centrality indices, the performance in terms of AUROC score decreases. However, not every addition of textual features leads to an increase in prediction performance; hence, understanding which features extract from textual content and performing a feature selection step, based on the network being studied, is important. In general, deep learning models are promising solutions even for the link prediction task with textual content but may suffer from the introduction of structured properties inferred from text.

## Data

For privacy reasons, it is not possibile to publish our data. To patch this problem, we provide an anonymized version of our data. They represent the final datasets for training the ML models.

## Code
We provide the code to run the experiments with the traditional (i.e. well know in the literature) supervised models, to construct the graph neural network (GNN) and graph autoencoder (GAE) architectures and use them on the link prediction task. They are available as jupyter notebooks to be easily executed and customized for other experiments.  

Since the way we compute features for link prediction is related to how we stored data retrieved from Steemit, we do not provide notebooks to compute the features.
Instead, we decided to publish some utility functions to compute them. They are available in python scripts. In this way you can re-use them with your graphs or collection of documents.

For data gathering you can refer to the [Steemit API](https://developers.steem.io/) documentation.

### Features for link prediction

``textual.py`` contains two utility funcitons:
- ``process_text_from_blockchain`` manipulates the list of comment\_op extracted from the Steemit Blockchain to obtain the data structures to compute text-based statistics
- ``get_user_interest_vectors`` computes the user interest vectors based on LDA topic distributions

For structural features, you can refer to the implementations available on [Networkx](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html)

### Link prediction with traditional supervised models

``Steemit-LinkPrediction-TraditionalClassifiers.ipynb`` contains the code to reproduce the experiments with the traditional supervised models. We used the implementations available on scikit-learn. The folder ``data/traditional/`` contains the data related to this part of the work.

### GNN and GAE models for link prediction
``SteemitGNN.ipynb`` and ``SteemitGAE.ipynb`` contain respectively the code to solve the link prediction task with GNN and GAE architectures. We used PyTorch Geometric to develop these models. The folder ``data/gnn/`` contains the data to fed both GNNs and GAEs. Specifically, the pickle files can be used to construct pyg datasets that represent both the structure of the graph and the text-based statistics as node attributes. We included GAE and GNN architectures and the functions to train and test them in two py files, namely ``gnn.py`` and ``gae.py``, just for the sake of clarity.

## Cite this work
```
@InProceedings{DileoDS22,
author="Dileo, Manuel
and Ba, Cheick Tidiane
and Zignani, Matteo
and Gaito, Sabrina",
editor="Pascal, Poncelet
and Ienco, Dino",
title="Link Prediction with Text in Online Social Networks: The Role of Textual Content on High-Resolution Temporal Data",
booktitle="Discovery Science",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="212--226",
abstract="Machine learning-based solutions for link prediction in Online Social Networks (OSNs) have been the subject of many research efforts. While most of them are mainly focused on the global and local properties of the graph structure surrounding links, a few take also into account additional contextual information, such as the textual content produced by OSN accounts. In this paper we cope with the latter solutions to i) evaluate the role of textual data in enhancing performances in the link prediction task on OSN; and ii) identify strengths and weaknesses of different machine learning approaches when dealing with properties extracted from text. We conducted the evaluation of several tools, from well-established methods such as logistic regression or ensemble methods to more recent deep learning architectures for graph representation learning, on a novel dataset gathered from an emerging blockchain online social network. This dataset represents a valuable playground for link prediction evaluation since it offers high-resolution temporal data on link creation and textual data for each account. Our findings show that the combination of structural and textual features enhances the prediction performance of traditional models. Deep learning architectures outperform the traditional ones and they can also benefit from the addition of textual features. However, some textual attributes can also reduce the prediction power of some deep architectures. In general, deep learning models are promising solutions even for the link prediction task with textual content but may suffer the introduction of structured properties inferred from the text.",
isbn="978-3-031-18840-4"
}
```
