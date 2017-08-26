# -*- coding: utf-8 -*-
"""
@author : Abhishek R S
"""

import numpy as np
import pandas as pd

# Import the dataset
dataset_train = pd.read_csv("training.csv")
dataset_test = pd.read_csv("testing.csv")
   
# Create X_train, X_test, Y_train and Y_test
X_train = dataset_train.iloc[:, 1:6].values
X_test = dataset_test.iloc[:, 1:6].values
Y_train = dataset_train.iloc[:, 0].values
Y_test = dataset_test.iloc[:, 0].values

# Preprocess the Data (Label encoding
from sklearn.preprocessing import LabelEncoder, StandardScaler
le_Y = LabelEncoder()
Y_train = le_Y.fit_transform(Y_train)
Y_test = le_Y.transform(Y_test)

# Scale the Data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Test the ANN on the Test Set
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Check the Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_pred, Y_test)
cm = confusion_matrix(Y_pred, Y_test)
cm