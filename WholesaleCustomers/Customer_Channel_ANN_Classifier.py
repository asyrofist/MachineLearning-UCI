
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('Wholesale customers data.csv')
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 0].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X[0]


# In[8]:


Y[0]


# ## Preprocess the Data

# In[9]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[10]:


le_Y = LabelEncoder()


# In[11]:


Y = le_Y.fit_transform(Y)


# In[12]:


Y[0]


# In[13]:


sc_X = StandardScaler()


# In[14]:


X = sc_X.fit_transform(X)


# In[15]:


X[0]


# ## Create Train and Test Data

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


Y_train.shape


# In[21]:


Y_test.shape


# In[22]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[23]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and train the ANN Classifier

# In[24]:


from keras.models import Sequential
from keras.layers import Dense


# In[25]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 6))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[26]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[27]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[28]:


accuracy_score(Y_test, Y_pred)


# In[29]:


confusion_matrix(Y_test, Y_pred)


# In[ ]:




