# -*- coding: utf-8 -*-
"""
@author : Abhishek R S
"""

import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Car.data", header = None)

# Creating X and Y
X = dataset.iloc[:, 0:6].values
Y = dataset.iloc[:, 6].values
                  
# Preprocess the Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)
ohey = OneHotEncoder(categorical_features=[0])
Y = ohey.fit_transform(Y).toarray()

le0 = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()

X[:, 0] = le0.fit_transform(X[:, 0])
X[:, 1] = le1.fit_transform(X[:, 1])
X[:, 2] = le2.fit_transform(X[:, 2])
X[:, 3] = le3.fit_transform(X[:, 3])
X[:, 4] = le4.fit_transform(X[:, 4])
X[:, 5] = le5.fit_transform(X[:, 5])

ohe0 = OneHotEncoder(categorical_features = [5])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

ohe0 = OneHotEncoder(categorical_features = [6])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

ohe0 = OneHotEncoder(categorical_features = [7])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

ohe0 = OneHotEncoder(categorical_features = [8])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

ohe0 = OneHotEncoder(categorical_features = [10])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

ohe0 = OneHotEncoder(categorical_features = [12])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

import keras
from keras.models import Sequential
from keras.layers import Dense

# For Y1 Prediction

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 15))

# Output Layer
clf_ann.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)

# Test the ANN on the Test Set
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Check the Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_pred, Y_test)