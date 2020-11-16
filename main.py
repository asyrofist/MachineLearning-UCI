import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
option = st.sidebar.selectbox('How would you like to be contacted?',['diagnosis', 'yeast', 'wine'])
dataset = pd.read_csv('data/'+option+'.data', header = None, delimiter = r"\s+", encoding = "utf-16")
hasil = dataset.head()
st.write(hasil)

X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values
st.write(X)
st.write(Y1)
st.write(Y2)

Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)
st.write(Y1)
st.write(Y2)
