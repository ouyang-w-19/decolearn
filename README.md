# decolearn

Decolearn is a Python implementation of ensemble learning algorithms described in the paper titled "LP-based Data Reduction for Diverse Ensemble Learning". This repository provides an implementation of algorithms from the paper, which is designed for (binary) classification problems.

##  System Requirements
Decolearn is tested on Python 3.8 (CPython) with following dependencies:

* gurobipy==9.5.1
* tensorflow==2.3.0
* keras==2.4.3
* matplotlib==3.5.1
* numpy==1.21.5
* scikit-learn==1.0.2

## Dataset
The test instances for binary classification were obtained from the widely-used CIFAR-10 dataset [1]. This dataset contains 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The test batch contains 1,000 randomly-selected images from each class, while the training batches contain the remaining images in random order. Some training batches may have an imbalance in the number of images per class.

For the experiments in this repository, two pairs of labels from CIFAR-10 were chosen with varying degrees of similarity. The label pair of categories 3 and 8 is considered easy to distinguish, while the label pair 3 and 2 requires more sophisticated learners to generate an ensemble that performs better than random.

## Testing Example
decolearn/run.py presents testing examples of solving binary classification problems on the mentioned Dataset.

## References
[1] Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

For more details about the algorithms and experimental setup, please refer to the paper "LP-based Data Reduction for Diverse Ensemble Learning".
