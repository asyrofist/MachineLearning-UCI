
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_excel('Blogger.xlsx')
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:, 5].values


# ## Preprocess the Data

# In[5]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)


# In[6]:


Y


# In[7]:


le0 = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()

X[:, 0] = le0.fit_transform(X[:, 0])
X[:, 1] = le1.fit_transform(X[:, 1])
X[:, 2] = le2.fit_transform(X[:, 2])
X[:, 3] = le_Y.transform(X[:, 3])
X[:, 4] = le_Y.transform(X[:, 4])


# In[8]:


X[0]


# In[9]:


ohe0 = OneHotEncoder(categorical_features = [0])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]


# In[10]:


ohe1 = OneHotEncoder(categorical_features = [2])
X = ohe1.fit_transform(X).toarray()
X = X[:, 1:]


# In[11]:


ohe1 = OneHotEncoder(categorical_features = [4])
X = ohe1.fit_transform(X).toarray()
X = X[:, 1:]


# ## Create Train and Test Data

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# ## Create and Train the Classifier

# In[13]:


from keras.models import Sequential
from keras.layers import Dense


# In[14]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 10))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Data
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[15]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[16]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test, Y_pred)


# In[17]:


confusion_matrix(Y_test, Y_pred)

