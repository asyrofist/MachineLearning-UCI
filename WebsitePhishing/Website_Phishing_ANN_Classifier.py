
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

# In[9]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[10]:


le_Y = LabelEncoder()


# In[11]:


Y = le_Y.fit_transform(Y)
Y


# In[12]:


Y = Y.reshape(len(Y), 1)
ohe_Y = OneHotEncoder(categorical_features = [0])


# In[13]:


Y = ohe_Y.fit_transform(Y).toarray()
Y


# In[14]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[15]:


for x in range(0, 9):
    encoder_X(x)


# In[16]:


X


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[18]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[19]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[20]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[21]:


ohe_X = OneHotEncoder(categorical_features = [8])


# In[22]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[23]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[24]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[25]:


ohe_X = OneHotEncoder(categorical_features = [10])


# In[26]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[27]:


ohe_X = OneHotEncoder(categorical_features = [11])


# In[28]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[29]:


ohe_X = OneHotEncoder(categorical_features = [12])


# In[30]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# ## Create Train and Test Data

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[33]:


X_train.shape


# In[34]:


X_test.shape


# In[35]:


Y_train.shape


# In[36]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[37]:


from keras.models import Sequential
from keras.layers import Dense


# In[38]:


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


# In[39]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[40]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[41]:


accuracy_score(Y_pred_class, Y_test_class)


# In[42]:


confusion_matrix(Y_pred_class, Y_test_class)


# In[ ]:




