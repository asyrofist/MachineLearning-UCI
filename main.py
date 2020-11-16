import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
with open("AcuteInflammations/diagnosis.data", encoding="utf8", errors='ignore') as dataset:
  X = dataset.iloc[:, 0:6].values
  Y1 = dataset.iloc[:, 6:7].values
  Y2 = dataset.iloc[:, 7:8].values
  st.dataframe(X)
