import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense


# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("Importing Data")
file_data = st.sidebar.selectbox('How would you like to be contacted?',['diagnosis', 'yeast', 'wine'])
dataset = pd.read_csv('data/'+file_data+'.data', header = None, delimiter = r"\s+", encoding = "utf-16")
hasil = dataset.head()
st.write(hasil)

# variable
start_X, end_X = st.sidebar.select_slider('Select X range?',options=[0, 1, 2, 3, 4, 5, 6, 7, 8],value=(0, 6))
start_Y1, end_Y1 = st.sidebar.select_slider('Select Y1 range?',options=[0, 1, 2, 3, 4, 5, 6, 7, 8],value=(end_X, end_X+1))
start_Y2, end_Y2 = st.sidebar.select_slider('Select Y2 range?',options=[0, 1, 2, 3, 4, 5, 6, 7, 8],value=(end_Y1, end_Y1+1))
X = dataset.iloc[:, start_X:end_X].values
Y1 = dataset.iloc[:, start_Y1:end_Y1].values
Y2 = dataset.iloc[:, start_Y2:end_Y2].values
Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)


#X Parameter
st.write(X)

#Y1 Parameter
col3, col4 = st.beta_columns([3,1])
pic1, ax = plt.subplots()
ax.hist(Y1, bins=20)
col3.pyplot(pic1)
col4.write(Y1)

# X Parameter
col5, col6 = st.beta_columns([3,1])
pic2, ax = plt.subplots()
ax.hist(Y2, bins=20)
col5.pyplot(pic2)
col6.write(Y2)

st.header("Preprocess the Data")
le_Y = LabelEncoder()
Y1 = le_Y.fit_transform(Y1)
Y2 = le_Y.transform(Y2)

col6, col7 = st.beta_columns([3,1])
pic3, ax = plt.subplots()
ax.hist(Y1, bins=20)
col6.pyplot(pic3)
col7.write(Y1)

col8, col9 = st.beta_columns([3,1])
pic4, ax = plt.subplots()
ax.hist(Y2, bins=20)
col8.pyplot(pic4)
col9.write(Y2)

le_X = LabelEncoder()
le_X.fit(X[:, 1])

X[:, 1] = le_X.transform(X[:, 1])
X[:, 2] = le_X.transform(X[:, 2])
X[:, 3] = le_X.transform(X[:, 3])
X[:, 4] = le_X.transform(X[:, 4])
X[:, 5] = le_X.transform(X[:, 5])
st.write(X)

st.header("Create and Train the Classifier for Y1")
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 4)

# model
clf_ann = Sequential()
clf_ann.add(Dense(3, activation = 'relu', kernel_initializer='glorot_uniform', input_dim = 6)) # First Hidden Layer
clf_ann.add(Dense(1, activation = 'sigmoid', kernel_initializer='glorot_uniform')) # Output Layer
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile the ANN
clf_ann.fit((X1_train).astype(np.float32), (Y1_train).astype(np.float32), batch_size = 5, epochs = 200) # Train the ANN on the Training Set

# #prediction
# Y1_pred = clf_ann.predict(X1_test) # Test the ANN on the Test Data
# Y1_pred = (Y1_pred > 0.5)
# akurasi = accuracy_score(Y1_test, Y1_pred)
# st.write(akurasi)
