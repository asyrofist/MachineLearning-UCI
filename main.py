import streamlit as st
import numpy as np
import pandas as pd
import StringIO

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
dataset = pd.read_csv("diagnosis.data", header = None, delimiter = r"\s+", encoding = "utf-16")
hasil = dataset.head()
st.write(hasil)
