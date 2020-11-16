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
with io.open("AcuteInflammations/diagnosis.data", 'r', encoding='utf-8') as fn:
  dataset = fn.readlines()
st.dataframe(dataset)
