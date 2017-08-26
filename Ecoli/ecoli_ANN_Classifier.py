
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


df = pd.read_csv('ecoli.data', header = None, delimiter = r"\s+")
df.shape


# In[3]:


df.head()


# ## Create X and Y

# In[4]:


Y = df.iloc[:, 8].values
Y


# In[5]:


X = df.iloc[:, 1:8].values
X


# ## Preprocess the Data

# In[6]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# In[7]:


le_Y = LabelEncoder()


# In[8]:


Y = le_Y.fit_transform(Y)
Y


# In[9]:


Y = Y.reshape(len(Y), 1)
ohe_Y = OneHotEncoder(categorical_features = [0])
Y = ohe_Y.fit_transform(Y).toarray()
Y


# In[10]:


sc_X = StandardScaler()


# In[11]:


X = sc_X.fit_transform(X)
X


# ## Create Train and Test Data

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[13]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[14]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and Train the Classifier

# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# In[20]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 7))

# Output Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[21]:


Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


accuracy_score(Y_pred, Y_test)


# In[ ]:




