# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:11:47 2023

@author: Bionformatic unit of IMIBIC
"""

## Load libraries
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from os import getcwd
import seaborn as sns
import warnings
import glob
warnings.filterwarnings('ignore')

## Input data
# =============================================================================
filename = glob.glob("./*.csv")[0]
dataset = pd.read_csv(filename, sep = ";", decimal = ",")
dataset["Phenotype"].replace(["LOW GRADE", "HIGH GRADE"], [0, 1], inplace = True)

feature_list = dataset.columns
dataset_df = dataset.corr(method='pearson')
sns.heatmap(dataset_df, annot=True) # Create correlation matrix

X = dataset.iloc[:, 0:4].values  
y = dataset.iloc[:, -1].values
    
## Scale the dataset
# =============================================================================
sc_X = StandardScaler()
X = sc_X.fit_transform(X)  
desc_pd = pd.DataFrame(X, columns = dataset.columns[:-2]).describe().round(4)

## Split dataset in training and test
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=dataset[["Phenotype"]], random_state = 1234)

classifier = KNeighborsClassifier(n_neighbors = 3, metric = "minkowski", p = 2)
classifier.fit(X_train, y_train)    
 
# Predict the test set
# =============================================================================
y_pred = classifier.predict(X_test)

## Create a confusion matrix and other parameters
# =============================================================================

cm_KNN = confusion_matrix(y_test, y_pred)
accuracy_KNN = accuracy_score(y_test, y_pred)

recall = np.diag(cm_KNN) / np.sum(cm_KNN, axis = 1)
mean_recall = np.mean(recall)

precision_KNN = precision_score(y_test, y_pred)
mean_precision = np.mean(precision_KNN)

kappa_KNN = cohen_kappa_score(y_test, y_pred)

sensitivity_KNN = recall_score(y_test, y_pred)
specificity_KNN = recall_score(y_test, y_pred, pos_label = 0)

results_pd = pd.DataFrame([accuracy_KNN,kappa_KNN,precision_KNN,
              sensitivity_KNN, specificity_KNN])

results_pd.columns = ["KNN"]
rownames_pd = ["Accuracy", "kappa", "Precision", "Sensitivity", "Specificity"]
results_pd.index = rownames_pd

## Print the results
# =============================================================================
print("\n--------------------------------------------------------------\n")
print("Correlation matrix:\n")
print(dataset_df)
print("\n--------------------------------------------------------------\n")
print("Confusion matrix:\n")
print(pd.DataFrame(cm_KNN, index = ["LOW GRADE", "HIGH GRADE"], columns = ["LOW GRADE", "HIGH GRADE"]))
print("\n--------------------------------------------------------------\n")
print("Statistic parameters:\n")
print(results_pd)

## Save the model in a new file
# =============================================================================
cwd = getcwd()
final_model = 'final_model.sav'
path = cwd + "\\" + final_model
print("\nModel was saved in", path)
dump(classifier, open(final_model, 'wb'))
print("--------------------------------------------------------------")