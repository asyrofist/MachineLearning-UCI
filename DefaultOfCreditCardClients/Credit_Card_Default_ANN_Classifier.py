# -*- coding: utf-8 -*-
"""
@author : Abhishek R S
"""

import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_excel("Dataset.xls")

# Creating X and Y
X = dataset.iloc[:, 0:23].values
Y = dataset.iloc[:, 23].values

# Preprocess the Data
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
le_2 = LabelEncoder()
le_3 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
X[:, 2] = le_2.fit_transform(X[:, 2])
X[:, 3] = le_3.fit_transform(X[:, 3])

ohe1 = OneHotEncoder(categorical_features = [1])
X = ohe1.fit_transform(X).toarray()
X = X[:, 1:]

ohe2 = OneHotEncoder(categorical_features = [2])
X = ohe2.fit_transform(X).toarray()
X = X[:, 1:]

ohe3 = OneHotEncoder(categorical_features = [8])
X = ohe3.fit_transform(X).toarray()
X = X[:, 1:]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Create Train and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

import keras
from keras.models import Sequential
from keras.layers import Dense

# For Y1 Prediction

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 30))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#clf_ann.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#clf_ann.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

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