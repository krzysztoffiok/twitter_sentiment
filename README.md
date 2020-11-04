# Analysis of Twitter sentiment with various Language Models
This repository contains data, code and results for fine-grained sentiment analysis. The paper describing this research is in preparation.

*"Analysis of sentiment in tweets addressed to a single Twitter account: comparison of model performance and explainability of predictions"*

Authors: Krzysztof Fiok1, Waldemar Karwowski1, Edgar Gutierrez-Franco12, and Maciej Wilamowski3


1 University of Central Florida, Department of Industrial Engineering & Management Systems, Orlando, Florida, USA </br>
2 2	Center for Latin-American Logistics Innovation, LOGyCA, Bogota, Colombia,110111.</br>
3 University of Warsaw, Faculty of Economic Sciences, Warsaw, Poland</br>
<br/>

The whole repository is published under MIT License (please refer to the [License file](https://github.com/krzysztoffiok/twitter_sentiment/blob/master/LICENSE)).

The code is written in Python3 and requires GPU computing machine for achieving reasonable performance.

## Results

sentiment@USNavy data set
FE Model | MMAE | MAE | EMD | MCC | F1 macro
-- | -- | -- | -- | -- | --
Term Frequency | 0.832 | 0.392 | 0.247 | 0.466 | 0.377
LIWC | 0.783 | 0.394 | 0.145 | 0.466 | 0.389
SEANCE | 0.756 | 0.377 | 0.155 | 0.479 | 0.395
Pooled FastText | 0.783 | 0.373 | 0.2 | 0.487 | 0.379
Pooled RoBERTa | 0.681 | 0.328 | 0.18 | 0.541 | 0.449
Universal Sentence Encoder | 0.701 | 0.325 | 0.182 | 0.541 | 0.41
FastText LSTM | 0.701 | 0.353 | 0.114 | 0.522 | 0.445
Roberta large LSTM | 0.536 | 0.297 | 0.058 | 0.588 | 0.561
Fine-tuned  RoBERTa large | 0.507 | 0.278 | 0.05 | 0.615 | 0.587
Fine-tuned BERT large uncased | 0.617 | 0.302 | 0.088 | 0.588 | 0.518
Fine-tuned BERT large cased | 0.609 | 0.297 | 0.097 | 0.585 | 0.491
Fine-tuned XLNet large cased | 0.582 | 0.288 | 0.089 | 0.598 | 0.525
Fine-tuned Bart large cnn | 0.5 | 0.268 | 0.051 | 0.626 | 0.596
Fine-tuned XLM-R | 0.515 | 0.269 | 0.062 | 0.626 | 0.592
Fine-tuned XLM MLM en 2048 | 0.594 | 0.281 | 0.085 | 0.604 | 0.496

SemEval2017 data set (Task 4 CE on data released only for that competition)
FE Model | MMAE | MAE | EMD | MCC | F1 macro
-- | -- | -- | -- | -- | --
Term Frequency | 1.372 | 0.723 | 0.699 | 0.046 | 0.138
LIWC | 1.167 | 0.617 | 0.51 | 0.167 | 0.215
SEANCE | 1.133 | 0.601 | 0.497 | 0.163 | 0.224
Pooled FastText | 1.129 | 0.595 | 0.507 | 0.2 | 0.228
Pooled RoBERTa | 1.075 | 0.576 | 0.505 | 0.23 | 0.248
Universal Sentence Encoder | 1.01 | 0.55 | 0.478 | 0.225 | 0.246
FastText LSTM | 1.013 | 0.559 | 0.393 | 0.213 | 0.275
Roberta large LSTM | 0.763 | 0.496 | 0.301 | 0.308 | 0.382
Fine-tuned  RoBERTa large | 0.662 | 0.452 | 0.265 | 0.344 | 0.417
Fine-tuned BERT large uncased | 0.863 | 0.483 | 0.316 | 0.294 | 0.34
Fine-tuned BERT large cased | 0.798 | 0.472 | 0.236 | 0.311 | 0.365
Fine-tuned XLNet large cased | 0.753 | 0.47 | 0.295 | 0.317 | 0.393
Fine-tuned Bart large cnn | 0.685 | 0.455 | 0.256 | 0.337 | 0.406
Fine-tuned XLM-R | 0.656 | 0.458 | 0.268 | 0.333 | 0.433
Fine-tuned XLM MLM en 2048 | 0.732 | 0.47 | 0.281 | 0.314 | 0.395

Abbrevations stand for:</br>
FE model: Feature Extraction model</br>
LIWC: Linguistic Enquiry and Word Count</br>
SEANCE: Sentiment analysis and social cognition engine</br>
MMAE: class-weighted MAE as in SemEval2017 (https://arxiv.org/abs/1912.00741)</br>
MAE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html</br>
EMD: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html</br>
MCC: sklearn.metrics.matthews_corrcoef</br>
F1: sklearn.metrics.f1_score(average="macro")</br>


## How to reproduce our paper results:
Please clone this repository and execute 4 bash scripts to repeat our experiments: <br>
usnavy_experiment.sh <br>
semeval_experiment.sh <br>
roberta_mbs_optim.sh <br>
LDA_usnavy.sh <br>

If you wish to carry out the LDA parameter grid search please use LDA_gensim_usnavy.ipynb.

## Acknowledgment
This research was carried out as part of the N000141812559 ONR research grant.

## Citation:<br/>
If you decide to use here published code or our dataset please cite our work in the following manner:
(please contact us directly at this time since the paper is still in preparation).

