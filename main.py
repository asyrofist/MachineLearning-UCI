import numpy as np
import pandas as pd

# dataset
dataset = pd.read_csv("diagnosis.data", header = None, delimiter = r"\s+")
X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values
a = dataset.head()
st.write(a)
