
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv("Car.data", header = None)
dataset.head()


# ## Create X and Y

# In[3]:


X = dataset.iloc[:, 0:6].values
Y = dataset.iloc[:, 6].values


# In[4]:


X.shape


# In[5]:


Y.shape


# In[6]:


X


# In[7]:


Y


# ## Preprocess the Data

# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)

def encoder(index):
    le = LabelEncoder()
    X[:, index] = le.fit_transform(X[:, index])


# In[9]:


for i in range(0, 6):
    encoder(i)


# In[10]:


X


# In[11]:


Y


# In[12]:


ohe_Y = OneHotEncoder(categorical_features = [0])
Y = ohe_Y.fit_transform(Y).toarray()


# In[13]:


Y


# In[14]:


ohe_X = OneHotEncoder(categorical_features = [5])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[15]:


ohe_X = OneHotEncoder(categorical_features = [6])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[16]:


ohe_X = OneHotEncoder(categorical_features = [7])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [8])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [10])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[19]:


ohe_X = OneHotEncoder(categorical_features = [12])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[20]:


X.shape


# ## Create Train and Test Data

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# ## Create and Train the Classifier

# In[22]:


from keras.models import Sequential
from keras.layers import Dense
clf_ann = Sequential()


# In[23]:


# First Hidden Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 15))

# Output Layer
clf_ann.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[24]:


Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_pred_class, Y_test_class)


# In[26]:


confusion_matrix(Y_pred_class, Y_test_class)


# In[ ]:




