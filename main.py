import streamlit as st
import numpy as np
import pandas as pd
import io

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
with open("AcuteInflammations/diagnosis.data", 'rb') as f:
  dataset = f.read(header = None, delimiter = r"\s+")
st.write(dataset)
