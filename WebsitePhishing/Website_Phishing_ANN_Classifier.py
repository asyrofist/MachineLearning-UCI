
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('PhishingData.csv', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:9].values
Y = dataset.iloc[:, 9].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X


# In[8]:


Y


# ## Preprocess the Data

# In[10]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[11]:


le_Y = LabelEncoder()


# In[12]:


Y = le_Y.fit_transform(Y)
Y


# In[13]:


Y = Y.reshape(len(Y), 1)
ohe_Y = OneHotEncoder(categorical_features = [0])


# In[14]:


Y = ohe_Y.fit_transform(Y).toarray()
Y


# In[15]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[16]:


for x in range(0, 9):
    encoder_X(x)


# In[17]:


X


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[19]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[21]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[22]:


ohe_X = OneHotEncoder(categorical_features = [8])


# In[23]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[24]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[25]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[26]:


ohe_X = OneHotEncoder(categorical_features = [10])


# In[27]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


ohe_X = OneHotEncoder(categorical_features = [11])


# In[29]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[30]:


ohe_X = OneHotEncoder(categorical_features = [12])


# In[31]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# ## Create Train and Test Data

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[34]:


X_train.shape


# In[35]:


X_test.shape


# In[36]:


Y_train.shape


# In[37]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[38]:


from keras.models import Sequential
from keras.layers import Dense


# In[51]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 16))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[52]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


accuracy_score(Y_pred, Y_test)


# In[ ]:




