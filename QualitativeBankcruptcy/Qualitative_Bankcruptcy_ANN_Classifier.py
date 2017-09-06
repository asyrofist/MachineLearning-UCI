
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('Qualitative_Bankruptcy.data.txt', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:6].values
Y = dataset.iloc[:, 6].values


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


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[13]:


for i in range(0, 6):
    encoder_X(i)


# In[14]:


X[0]


# In[15]:


ohe_X = OneHotEncoder(categorical_features = [5])


# In[16]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


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


# ## Create Train and Test Data

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# In[31]:


Y_train.shape


# In[32]:


Y_test.shape


# In[33]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[34]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and train the ANN Classifier

# In[35]:


from keras.models import Sequential
from keras.layers import Dense


# In[36]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))

# Second Hidden Layer
#clf_ann.add(Dense(output_dim = , init = 'uniform', activation = 'relu'))

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




