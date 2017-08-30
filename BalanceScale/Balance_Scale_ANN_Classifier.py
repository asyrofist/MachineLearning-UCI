
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv("balance-scale.data", header = None)


# In[3]:


X = dataset.iloc[:, 1:5].values
Y = dataset.iloc[:, 0].values


# In[4]:


dataset.head()


# ## Preprocess the Data

# In[5]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)


# In[6]:


ohe = OneHotEncoder(categorical_features=[0])
Y = ohe.fit_transform(Y).toarray()


# In[7]:


Y


# In[8]:


sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[9]:


X


# ## Create Train and Test Data

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# ## Create and Train the Classifier

# In[11]:


from keras.models import Sequential
from keras.layers import Dense

clf_ann = Sequential()


# In[12]:


# First Hidden Layer
clf_ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 4))

# Output Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[13]:


# Test the Classifier on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_pred_class, Y_test_class)


# In[15]:


confusion_matrix(Y_pred_class, Y_test_class)


# In[ ]:




