
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
Y = dataset.iloc[:, 1].values


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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[10]:


le_Y = LabelEncoder()


# In[11]:


Y = le_Y.fit_transform(Y)
Y[0]


# In[12]:


Y = Y.reshape(len(Y), 1)
ohe_Y = OneHotEncoder(categorical_features = [0])


# In[13]:


Y = ohe_Y.fit_transform(Y).toarray()
Y[0]


# In[14]:


sc_X = StandardScaler()


# In[15]:


X = sc_X.fit_transform(X)


# In[16]:


X[0]


# ## Create Train and Test Data

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


Y_train.shape


# In[22]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[23]:


from keras.models import Sequential
from keras.layers import Dense


# In[24]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 6))

# Output Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[25]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[26]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[27]:


accuracy_score(Y_test_class, Y_pred_class)


# In[28]:


confusion_matrix(Y_test_class, Y_pred_class)


# In[ ]:




