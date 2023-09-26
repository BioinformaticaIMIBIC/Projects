# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:22:13 2023

@author:  Bionformatic unit of IMIBIC
"""

## Load libraries
# =============================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import numpy as np


## Input data
# =============================================================================
test_filename = glob.glob("./*.csv")[0]
dataset = pd.read_csv(test_filename, sep = ";", decimal = ",")
dataset["Phenotype"].replace(["LOW GRADE", "HIGH GRADE"], [0, 1], inplace = True)

feature_list = dataset.columns


X_test = dataset.iloc[:, 0:4].values  


## Scale the dataset
# =============================================================================
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)

## Load model and predict
# =============================================================================
model = glob.glob("./*.sav")[0]
loaded_model = pickle.load(open(model, 'rb'))

results = pd.DataFrame(loaded_model.predict(X_test))

definitions =  ["LOW GRADE", "HIGH GRADE"]
reversefactor = dict(zip(range(2),definitions))
results = np.vectorize(reversefactor.get)(results)

final_results = pd.DataFrame(dataset.iloc[:, 0:4], columns = dataset.columns[0:4])
final_results.insert(4, "Phenotype", results)

## Print the results
# =============================================================================
print("\n--------------------------------------------------------------\n")
print("Prediction:\n")
print(final_results)
print ("--------------------------------------------------------------")
