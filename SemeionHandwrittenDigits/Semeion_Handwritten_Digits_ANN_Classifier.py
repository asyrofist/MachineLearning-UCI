
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('semeion.data', header = None, delimiter = r"\s+")
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:256].values
Y = dataset.iloc[:, 256:].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X[0]


# In[8]:


Y[0]


# ## Create Train and Test Data

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


Y_train.shape


# In[14]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# In[16]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 133, init = 'uniform', activation = 'relu', input_dim = 256))

# Second Hidden Layer
#clf_ann.add(Dense(output_dim = 133, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[17]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[18]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[19]:


accuracy_score(Y_test_class, Y_pred_class)


# In[20]:


confusion_matrix(Y_test_class, Y_pred_class)


# In[ ]:




