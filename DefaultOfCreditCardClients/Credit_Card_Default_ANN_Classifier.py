
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_excel('Dataset.xls')
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


Y = dataset.iloc[:, 23].values
Y.shape


# In[5]:


Y[0]


# In[6]:


X = dataset.iloc[:, 0:23].values
X.shape


# In[7]:


X[0]


# ## Preprocess the Data

# In[8]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# In[9]:


le_X = LabelEncoder()
X[:, 1] = le_X.fit_transform(X[:, 1])


# In[10]:


le_X = LabelEncoder()
X[:, 2] = le_X.fit_transform(X[:, 2])


# In[11]:


le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])


# In[12]:


ohe_X = OneHotEncoder(categorical_features = [1])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[13]:


ohe_X = OneHotEncoder(categorical_features = [2])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[14]:


ohe_X = OneHotEncoder(categorical_features = [8])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[15]:


sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[16]:


pd.DataFrame(X).head()


# ## Create Train and Test Data

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[18]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[19]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and train the Classifier

# In[20]:


from keras.models import Sequential
from keras.layers import Dense


# In[21]:


clf_ann = Sequential()

clf_ann.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 30))

clf_ann.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))

clf_ann.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[22]:


Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[24]:


accuracy_score(Y_test, Y_pred)


# In[25]:


confusion_matrix(Y_test, Y_pred)


# In[ ]:




