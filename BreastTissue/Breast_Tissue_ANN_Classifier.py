
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


df = pd.read_excel('BreastTissue.xls', sheetname = 1)
df.shape


# In[3]:


df.head()


# ## Create X and Y

# In[4]:


X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values


# ## Preprocess the Data

# In[5]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[6]:


X[0]


# In[7]:


le_Y = LabelEncoder()


# In[8]:


Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)
Y.shape


# In[9]:


ohey = OneHotEncoder(categorical_features = [0])


# In[10]:


Y = ohey.fit_transform(Y).toarray()
Y[0]


# ## Create Train and Test Data

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# ## Create and Train the Classifier

# In[12]:


from keras.models import Sequential
from keras.layers import Dense


# In[13]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = 9))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Training Set
clf_ann.fit(X_train, Y_train, batch_size = 5, nb_epoch = 200)


# In[14]:


Y_pred = clf_ann.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_test_class = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[15]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test_class, Y_pred_class)


# In[16]:


confusion_matrix(Y_test_class, Y_pred_class)


# In[ ]:




