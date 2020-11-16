import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
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
st.dataframe(X, Y1, Y2)

Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)
# st.write(Y1)
# st.write(Y2)
