# Predicting-the-critical-temperature-of-superconductors
Prediction of critical temperature of superconductors by using random forest and neural networks
## Super conductors
A superconductor is a material that can conduct electricity or transport electrons from one atom to another with no resistance.This means no heat, sound or any other form of energy would be released from the material when it has reached "critical temperature" (Tc)

## Critical Temperature
The temperature at which the material becomes superconductive. Most materials must be in an extremely low energy state (very cold) in order to become superconductive.

## Goal
The goal is to predict critical temperature of the given material based on different features. The soal aim is predict Tc but not wheter the material is superconductor or not.

## Report
Please refer Predicting the Critical Temperature of a Superconductor.pdf in this repository for the full implemention of the models and to study how I apporached to final predictions.

### Prerequisites
```
Rstudio 
```

## Getting the data

[Dataset](http://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) - Link to website where dataset is available.

It is also available in this repository named as train.csv and unique_m.csv


## Datapreprocessing
* remove or replace na elements if availble
* split the datainto train and test set
* scale the data

## Models used
* Random Forest Regression
* Neural Networks

## Tuning the models
* For random forest done grid serach for finding hyperparmeters like mtry, nodesize, ntree, sample size by using ranger(which is faster implementaion of random forest)
* For Neural Networks maually tested on using different layers and units and choosed the best parameters.

## steps to follow
* load the rscript file ```load("Predict_Tc_script.R")```
* load sc_tc.RData ```load("sc_tc.RData")```
```
install.packages("CHNOSZ")
library(CHNOSZ)
```
* for neural network
```
install.packages("keras")
library(keras)
install.packages("reticulate")
library(reticulate)
```
* this is most important ```reticulate::use_python('set your python path')```
* to know your python in your computer type ```which python```
* now you will be in python shell from r. type ```import tensorflow```
* exit the shell

## Start predicting the temperature by matreial 
```
predict_tc_rf("Sr0.1La1.9Cu1O4")  use this to predict by using random forest
predict_tc_nn("Sr0.1La1.9Cu1O4")  use this to predict by using neural network
```
in the place of Sr0.1La1.9Cu1O4 you can also use like Ba0.2La1.8Cu1O4 etc..,
## Output
Even though both are good at predicting the critical temperature, Neural Network outperformed random forest.

#### Prediction for material Sr0.1La1.9Cu1O4 
The actual critical Temperature for Sr0.1La1.9Cu1O4 is 33

random forest | neural network
------------ | -------------
48| 30

For some materails random forest prediction is not so good but this is not the case for other materials, whereas neural network made good prediction close to actual values.

#### Neural network plots for loss and mse
![Plot for neural network](https://github.com/chandravamshi/Predicting-the-critical-temperature-of-superconductors/blob/master/plots/nn-final-.png)

#### random forest plots for out-of-bag and validation error
![Plot for rf_oob_valid](https://github.com/chandravamshi/Predicting-the-critical-temperature-of-superconductors/blob/master/plots/rf_oob_valid_error.png)




