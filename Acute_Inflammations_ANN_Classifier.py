# -*- coding: utf-8 -*-
"""
@author : Abhishek R S
"""

import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv("diagnosis.data", header = None, delimiter = r"\s+")
X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values
                  
Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)
                
# Preprocess the Data (Label encoding
from sklearn.preprocessing import LabelEncoder
le_Y = LabelEncoder()
Y1 = le_Y.fit_transform(Y1)
Y2 = le_Y.transform(Y2)

# Scale the Data
le_X = LabelEncoder()
le_X.fit(X[:, 1])

X[:, 1] = le_X.transform(X[:, 1])
X[:, 2] = le_X.transform(X[:, 2])
X[:, 3] = le_X.transform(X[:, 3])
X[:, 4] = le_X.transform(X[:, 4])
X[:, 5] = le_X.transform(X[:, 5])

from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 4)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 4)

import keras
from keras.models import Sequential
from keras.layers import Dense

# For Y1 Prediction

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 6))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X1_train, Y1_train, batch_size = 5, nb_epoch = 200)

# Test the ANN on the Test Set
Y1_pred = clf_ann.predict(X1_test)
Y1_pred = (Y1_pred > 0.5)

# Check the Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y1_pred, Y1_test)
cm = confusion_matrix(Y1_pred, Y1_test)
cm

# For Y2 Prediction

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 6))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X2_train, Y2_train, batch_size = 5, nb_epoch = 200)

# Test the ANN on the Test Set
Y2_pred = clf_ann.predict(X2_test)
Y2_pred = (Y2_pred > 0.5)

# Check the Accuracy
accuracy_score(Y2_pred, Y2_test)
cm = confusion_matrix(Y1_pred, Y1_test)
cm