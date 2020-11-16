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
     spamreader = pd.read_csv(csvfile, delimiter= r"\s+")
#      for row in spamreader:
#          print ', '.join(row)
st.dataframe(spamreader)
