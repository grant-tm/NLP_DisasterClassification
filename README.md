# NLP_DisasterClassification
## Problem Description
Monitoring social media can quickly identify ongoing disasters, but when someone writes "... the sky was ablaze...", they may be referring to either a sunset or a devastation wildfire.
[This NLP practice problem](https://www.kaggle.com/competitions/nlp-getting-started/overview) asks you to train a model capable of distinguishing genuine disaster reports from inert hyperbole and metaphor.
## Materials
This repository contains a .csv file of predictions for the testing dataset and the Python program I used to generate them.
As per Kaggle guidelines, the training and testing datasets themselves are not included in this repository but are are freely available on the [problem page](https://www.kaggle.com/competitions/nlp-getting-started/overview).
## Solution
nlp.py is a Python program using the Keras API with a Tensorflow backend to train a small neural net on the problem's provided training dataset.
Running this program produces a .csv file of predictions for the test set which yields an f1 score of 0.8.
