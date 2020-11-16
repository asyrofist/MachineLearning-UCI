import streamlit as st
import numpy as np
import pandas as pd
import csv


# dataset
st.write("""
# Machine Learning
Berikut ini algoritma yang digunakan untuk Dataset UCI
""")

st.header("UCI Dataset")
with open("AcuteInflammations/diagnosis.data", 'rb') as csvfile:
     spamreader = pd.read_csv(csvfile, header = None, delimiter= r"\s+")
st.dataframe(spamreader)
