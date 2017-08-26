
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv("data_banknote_authentication.txt", header=None)


# In[4]:


dataset.columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]


# In[5]:


dataset.head()


# ## Creating X and Y

# In[6]:


X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values


# In[7]:


X


# In[8]:


Y


# ## Create Train and Test Data

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 4)


# In[11]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[10]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and Train the Classifier

# In[13]:


from keras.models import Sequential
from keras.layers import Dense


# In[15]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 4))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the Classifier on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[16]:


# Test the Classifier on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[17]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[18]:


accuracy_score(Y_pred, Y_test)


# In[19]:


confusion_matrix(Y_pred, Y_test)


# In[ ]:




