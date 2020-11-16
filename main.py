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
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     stringio = StringIO(uploaded_file.decode("utf-8"))
     st.write(stringio)
