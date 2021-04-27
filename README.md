===
Code for TBD

Introduction
---
Our research analysis is based on GraphSum [Li et. al](https://arxiv.org/pdf/2005.10043.pdf).

The git consists 2 branches namely RQ1 and RQ2.

RQ1-branch is able to reproduce the results for sentence vs paragraph comparsion for the MultiNews dataset. The RQ2-branch is able to reproduce our results for the source origin analysis.

In order to run our scripts, you first need to download the raw MultiNews data from [this link](https://github.com/Alex-Fabbri/Multi-News) and the ranked WikiSum dataset from [here](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum).

After obtaining those datasets you can pre-process the datasets for the specific research question. 

Namely for RQ1, you can run ./src/preprocess_multinews.sh, where you can modify the desired parameters.

For RQ2 you can run ./src/preprocess

