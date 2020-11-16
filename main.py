import streamlit as st
import numpy as np
import pandas as pd

# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
with open("AcuteInflammations/diagnosis.data",'rb') as f:
    dataset = f.read()
dataset = dataset.rstrip("\n").decode("utf-16")
dataset = dataset.split("\r\n")
st.dataframe(dataset)
