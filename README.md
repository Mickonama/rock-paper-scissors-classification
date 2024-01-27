# rock-paper-scissors-classification
Rock paper scissors classification using SVM, Random Forest and CNN

## Dependancies:
- Numpy
- Matplotlib
- Sklearn
- Tensroflow

## The dataset:
The main dataset can be found on [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). It contains images of hand gestures infront of a green screen that represent a rock, a paper or scissors.

The second dataset can be found [here](https://www.kaggle.com/datasets/yash811/rockpaperscissors/data).

## Implementation steps:

### Data augmentation

We flip the images on every axis (vertically, horizontally, vertically and horizontally). This way we cover every possible action that the random agent can take. Afterwards, we apply a Gaussian Noise on the images. We split the dataset into a train and test set with a 70-30% split.

### Dimensionality Reduction

We perform dimensionality reduction on the dataset using KPCA. This reduction will only be used for the classifiers that use 1D input arrays as samples (unlike CNN).
