# HSIC_sensitive_features

# A statistical approach to detect sensitive features in a group fairness setting

## Introduction

This work proposes an approach to detect sensitive features before training step. Our proposal is based on the Hilbert-Schmidt Independence Criterion (HSIC) and the hypothesis is that features with high dependence with the outcomes may entail disparate results. In order to evaluate our proposal, we consider the following datasets:

- Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
- COMPAS Recidivism risk: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- Taiwanese Default Credit: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
- LSAC: http://www.seaphe.org/databases.php
