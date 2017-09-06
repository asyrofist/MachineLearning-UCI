
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('tae.data', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:, 5].values


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


for x in range(0, X.shape[1] - 1):
    encoder_X(x)


# In[14]:


X[0]


# In[15]:


ohe_X = OneHotEncoder(categorical_features = [2])


# In[16]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [26])


# In[18]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[19]:


sc_X = StandardScaler()


# In[20]:


X = sc_X.fit_transform(X)
X


# In[21]:


Y = Y.reshape(len(Y), 1)
ohe_Y = OneHotEncoder(categorical_features = [0])


# In[22]:


Y = ohe_Y.fit_transform(Y).toarray()
Y[0]


# ## Create Train and Test Data

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


Y_train.shape


# In[28]:


Y_test.shape


# ## Create and train the ANN Classifier

# In[29]:


from keras.models import Sequential
from keras.layers import Dense


# In[30]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 27, init = 'uniform', activation = 'relu', input_dim = 52))

# Output Layer
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[31]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[32]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[33]:


accuracy_score(Y_test_class, Y_pred_class)


# In[34]:


confusion_matrix(Y_test_class, Y_pred_class)


# In[ ]:




