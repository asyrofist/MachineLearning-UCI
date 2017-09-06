
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv("Dataset.txt", header = None, delimiter = r"\s+")
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:18].values
Y = dataset.iloc[:, 18].values


# In[5]:


X[0]


# In[6]:


Y[0]


# ## Preprocess the Data

# In[7]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[8]:


le_Y = LabelEncoder()


# In[9]:


Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)
Y[0]


# In[10]:


ohe_Y = OneHotEncoder(categorical_features = [0])


# In[11]:


Y = ohe_Y.fit_transform(Y).toarray()
Y[0]


# In[12]:


sc_X = StandardScaler()


# In[13]:


X = sc_X.fit_transform(X)
X[0]


# ## Create Train and Test Data

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


Y_train.shape


# In[18]:


Y_test.shape


# ## Create and train the Classifier

# In[19]:


from keras.models import Sequential
from keras.layers import Dense


# In[20]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 18))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 200)


# In[21]:


Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test_class, Y_pred_class)


# In[23]:


confusion_matrix(Y_test_class, Y_pred_class)


# In[ ]:




