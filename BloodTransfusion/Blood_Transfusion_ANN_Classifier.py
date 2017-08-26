
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('transfusion.data')
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values


# ## Preprocess the Data

# In[5]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[6]:


X


# ## Create Train and Test Data

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[8]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[9]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and Train the Classifier

# In[10]:


from keras.models import Sequential
from keras.layers import Dense


# In[11]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 4))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Third Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Fourth Hidden Layer
clf_ann.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[12]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[13]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_pred, Y_test)


# In[14]:


confusion_matrix(Y_pred, Y_test)


# In[ ]:




