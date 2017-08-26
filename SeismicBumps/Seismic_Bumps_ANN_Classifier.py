
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('Seismic Bumps.csv', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:18].values
Y = dataset.iloc[:, 18].values


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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[10]:


le_Y = LabelEncoder()


# In[11]:


Y = le_Y.fit_transform(Y)
Y


# In[12]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[13]:


to_be_encoded_indices = [0, 1, 2, 7]


# In[14]:


for x in to_be_encoded_indices:
    encoder_X(x)


# In[15]:


X


# In[16]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[17]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [3])


# In[19]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


sc_X = StandardScaler()


# In[21]:


X = sc_X.fit_transform(X)
X


# ## Create Train and Test Data

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


Y_train.shape


# In[27]:


Y_test.shape


# In[28]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[29]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and train the ANN Classifier

# In[30]:


from keras.models import Sequential
from keras.layers import Dense


# In[46]:


clf_ann = Sequential()

# 10 20 40 80 160
# 40 40 160

# First Hidden Layer
clf_ann.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu', input_dim = 20))

# Second Hidden Layer
#clf_ann.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

clf_ann.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))

#clf_ann.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

clf_ann.add(Dense(output_dim = 160, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[47]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[48]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[49]:


accuracy_score(Y_pred, Y_test)


# In[50]:


confusion_matrix(Y_pred, Y_test)


# In[ ]:




