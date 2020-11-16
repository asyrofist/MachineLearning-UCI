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
nilai_list1 = st.multiselect('What are your favorite colors',[0, 1, 2, 3, 4, 5, 6, 7, 8],[0, 6])
nilai_list2 = st.multiselect('What are your favorite colors',[0, 1, 2, 3, 4, 5, 6, 7, 8],[6, 7])
nilai_list3 = st.multiselect('What are your favorite colors',[0, 1, 2, 3, 4, 5, 6, 7, 8],[7, 8])
st.write(nilai_list1, nilai_list2, nilai_list3)

X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values
st.write(X)
st.write(Y1)
st.write(Y2)


Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)
# st.write(Y1)
# st.write(Y2)
