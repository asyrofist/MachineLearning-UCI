
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('ThoraricSurgery.csv', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:16].values
Y = dataset.iloc[:, 16].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X[0]


# ## Preprocess the Data

# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[9]:


le_Y = LabelEncoder()


# In[10]:


Y = le_Y.fit_transform(Y)
Y[0]


# In[11]:


def enocder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[12]:


X_indices = [0] + list(range(3,15))
X_indices


# In[13]:


for x in X_indices:
    enocder_X(x)


# In[14]:


X.shape


# In[15]:


pd.DataFrame(pd.DataFrame(X[:, 9])[0].value_counts())


# In[16]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[17]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[18]:


X = X[:, 1:]
X.shape


# In[19]:


pd.DataFrame(pd.DataFrame(X[:, 6])[0].value_counts())


# In[20]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[21]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[22]:


X = X[:, 1:]
X.shape


# In[23]:


pd.DataFrame(pd.DataFrame(X[:, 5])[0].value_counts())


# In[24]:


ohe_X = OneHotEncoder(categorical_features = [5])


# In[25]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[26]:


X = X[:, 1:]
X.shape


# In[27]:


sc_X = StandardScaler()


# In[28]:


X = sc_X.fit_transform(X)
X[0]


# ## Create Train and Test Data

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# In[33]:


Y_train.shape


# In[34]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[35]:


from keras.models import Sequential
from keras.layers import Dense


# In[36]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu', input_dim = 24))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[37]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[38]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[39]:


accuracy_score(Y_test, Y_pred)


# In[40]:


confusion_matrix(Y_test, Y_pred)


# In[ ]:




