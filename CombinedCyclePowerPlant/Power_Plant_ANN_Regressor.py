
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_excel('Folds5x2_pp.xlsx')
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X


# In[8]:


Y


# ## Preprocess the Data

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc_X = StandardScaler()


# In[11]:


X = sc_X.fit_transform(X)


# In[12]:


X


# ## Create Train and Test Data

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


Y_train.shape


# In[18]:


Y_test.shape


# ## Create and train the ANN Regressor

# In[19]:


from keras.models import Sequential
from keras.layers import Dense


# In[20]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = 4))

# Second Hidden Layer
#clf_ann.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#clf_ann.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 100)


# In[21]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)


# ## Check the Regression Metrics

# In[22]:


from sklearn.metrics import mean_squared_error, r2_score


# In[23]:


mean_squared_error(Y_pred, Y_test)


# In[24]:


r2_score(Y_pred, Y_test)


# In[ ]:




