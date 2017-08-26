# -*- coding: utf-8 -*-
"""
@author : Abhishek R S
"""

import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv("iris.data", header = None)
   
# Create X and Y        
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4:5].values

# Preprocess the Data (Label encoding and One Hot Encoding) 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)
ohe = OneHotEncoder(categorical_features=[0])
Y = ohe.fit_transform(Y).toarray()

# Training and Test Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

# Scale the Data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 4))

# Output Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)

# Test the ANN on the Test Set
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Check the Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_pred, Y_test)