
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd
import keras


# ## Importing the Data

# In[2]:


dataset_train = pd.read_csv("training.csv")
dataset_test = pd.read_csv("testing.csv")


# In[3]:


dataset_test.head()


# In[4]:


dataset_train.head()


# In[5]:


dataset_test.shape


# In[6]:


dataset_train.shape


# ## Create X and Y

# In[7]:


X_train = dataset_train.iloc[:, 1:6].values
X_test = dataset_test.iloc[:, 1:6].values
Y_train = dataset_train.iloc[:, 0].values
Y_test = dataset_test.iloc[:, 0].values


# In[8]:


X_train.shape


# In[9]:


X_test.shape


# In[10]:


Y_train.shape


# In[11]:


Y_test.shape


# ## Preprocess the Data

# In[12]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
le_Y = LabelEncoder()
Y_train = le_Y.fit_transform(Y_train)
Y_test = le_Y.transform(Y_test)


# In[13]:


Y_train[0]


# In[14]:


Y_test[0]


# In[15]:


# Scale the Data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[16]:


X_train[0]


# In[17]:


X_test[0]


# ## Create and Train the Classifier

# In[18]:


from keras.models import Sequential
from keras.layers import Dense


# In[19]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[20]:


Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test, Y_pred)


# In[22]:


confusion_matrix(Y_test, Y_pred)


# In[ ]:




