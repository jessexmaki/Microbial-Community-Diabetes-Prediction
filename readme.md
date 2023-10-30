# Microbiome Disease Risk Prediction
This project develops machine learning models to predict disease risk from human microbiome sequence data.

## Overview
The repository contains code to train neural network models to predict type 1 diabetes status based on microbiome composition data. The models take microbiome 16S rRNA sequence profiles as input and output a prediction of whether the sample came from an individual with type 1 diabetes.

The goal is to identify microbial signatures associated with type 1 diabetes status that can allow accurate prediction of disease risk from sequence data alone.

### Data
The dataset used is from a published study of 16S rRNA sequencing on samples from 16,344 individuals. It contains relative abundance profiles of 256 bacterial OTUs (operational taxonomic units) in each sample.

Samples are labeled as either 'control' (no diabetes) or 'case' (type 1 diabetes). The dataset is split 80/20 into training and validation sets.

### Models
Several neural network architectures are implemented:

- Simple linear model
- 1D convolutional neural network
- LSTM recurrent neural network
  
Performance is evaluated based on accuracy of predicting disease status labels in the held-out validation set.

### Results

The simple linear model achieves approximately 78% accuracy on the validation set. This suggests most of the predictive signal is contained in a linear function of the input features.
The 1D convolutional neural network obtains over 81% validation accuracy after 50 epochs of training. The nonlinear activations and convolutional layers are able to extract slightly more predictive signal compared to the linear model.
The LSTM model reaches nearly 80% validation accuracy after 50 epochs. The sequential modeling capability enables it to identify temporal patterns predictive of disease status.
Overall, all the models achieve good performance with validation accuracy in the 78-81% range. This indicates the microbiome data does contain significant signals related to type 1 diabetes status that can be detected by machine learning models.
The models have not yet reached saturation - with further hyperparameter tuning and neural architecture search, even better accuracy may be possible.
No significant overfitting is observed during training, indicating the datasets are sufficiently large to train complex models.
