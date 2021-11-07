# ------
# Data Processing Script for "Identifying Predictors of Opioid Overdose Death with Machine Learning"
# Installing and loading necessary libraries
# ------


# Clear workspace
rm(list=ls())

# Installing and loading required libraries
install.packages(c("glmnet","glmnetUtils","dplyr","randomForest","ggplot2","imputeMissings","caret", lib = '~/Rlibs'))


library(glmnet)
library(glmnetUtils)
library(dplyr)
library(randomForest)
library(ggplot2)
library(imputeMissings)
library(caret)
library(ModelMetrics)
