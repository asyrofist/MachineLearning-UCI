import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
# dataset = pd.read_csv("AcuteInflammations/diagnosis.data", header = None, delimiter = r"\s+")
dataset = pd.read_csv("AcuteInflammations/diagnosis.data")
st.write(dataset)
