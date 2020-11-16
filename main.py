import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  dataset = pd.read_csv("AcuteInflammations/diagnosis.data", header = None, delimiter = r"\s+", encoding = "utf-16")
  hasil = dataset.head()
  st.write(hasil)
