Code for  [**Analysis of GraphSum’s Attention Weights to Improve the
Explainability of Multi-Document Summarization**](https://github.com/arnelochner/GBTBMDS/blob/main/scientific_report/Project_Data_Science___Text_Summarization___Scientific_Report.pdf).


Introduction
---
Our research analysis is based on GraphSum [Li et. al](https://arxiv.org/pdf/2005.10043.pdf) [ACL2020-GraphSum](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-GraphSum).

The git consists 2 branches namely RQ1 and RQ2.

RQ1-branch is able to reproduce the results for sentence vs paragraph comparsion for the MultiNews dataset. The RQ2-branch is able to reproduce our results for the source origin analysis.

Code 
--- 
In order to run our scripts, you first need to download the raw MultiNews data from [this link](https://github.com/Alex-Fabbri/Multi-News) and the ranked WikiSum dataset from [here](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum).

After obtaining those datasets you can pre-process the datasets for the specific research question. 

Namely for RQ1, you can run ./scripts/preprocess_multinews.sh, where you can modify the desired parameters.

For RQ2 you can run either ./scripts/preprocess_multinews.sh or ./scripts/preprocess_wikisum.sh.

Afterwards you can start training the GraphSum model on sentence and paragraph level with ./scripts/run_graphsum_local_multinews_sentence.sh and ./scripts/run_graphsum_local_multinews_paragraphs.sh respectively. The obtained rouge scores can then be found in the log folder.

For RQ2 you can start the code pipeline with ./scripts/rq2_multinews.sh and ./scripts/rq2_wikisum.sh, within those bashscripts you can set parameters like output directories and more.

Further documentation can be found in the code or our [technical report](https://github.com/arnelochner/GBTBMDS/blob/main/technical_report/Project_Data_Science___Text_Summarization___Technical_Report.pdf).

