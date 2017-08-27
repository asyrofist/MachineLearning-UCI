
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import keras


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('bank.csv', delimiter=';', quoting=3)
dataset.shape


# In[3]:


dataset.head()


# In[4]:


dataset = dataset.rename(columns = lambda x : x.replace('"', ''))
dataset.head()


# In[5]:


columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for x in columns:
    dataset[x] = dataset[x].apply(lambda x : x.replace('"', ''))


# In[6]:


dataset.head()


# In[7]:


dataset['age'] = pd.to_numeric(dataset['age'])


# ## Create X and Y

# In[8]:


X = dataset.iloc[:, 0:16].values
Y = dataset.iloc[:, 16].values


# In[9]:


X.shape


# In[10]:


Y.shape


# In[11]:


X


# In[12]:


Y


# ## Preprocess the Data

# In[13]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[14]:


le_Y = LabelEncoder()


# In[15]:


Y = le_Y.fit_transform(Y)
Y


# In[16]:


cols_to_encode = [1, 2, 3, 4, 6, 7, 8, 10, 15]
def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])
    return


# In[17]:


for i in cols_to_encode:
    encoder_X(i)


# In[18]:


X[0, :]


# In[19]:


ohe_X = OneHotEncoder(categorical_features = [15])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


pd.DataFrame(pd.DataFrame(X[:, 13])[0].value_counts())


# In[21]:


ohe_X = OneHotEncoder(categorical_features = [13])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[22]:


pd.DataFrame(pd.DataFrame(X[:, 22])[0].value_counts())


# In[23]:


ohe_X = OneHotEncoder(categorical_features = [22])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[24]:


pd.DataFrame(pd.DataFrame(X[:, 19])[0].value_counts())


# In[25]:


ohe_X = OneHotEncoder(categorical_features = [19])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[26]:


pd.DataFrame(pd.DataFrame(X[:, 21])[0].value_counts())


# In[27]:


ohe_X = OneHotEncoder(categorical_features = [21])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


pd.DataFrame(pd.DataFrame(X[:, 22])[0].value_counts())


# In[29]:


ohe_X = OneHotEncoder(categorical_features = [22])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[30]:


sc_X = StandardScaler()


# In[31]:


X = sc_X.fit_transform(X)


# In[32]:


X


# ## Create Train and Test Data

# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[35]:


X_train.shape


# In[36]:


X_test.shape


# In[37]:


Y_train.shape


# In[38]:


Y_test.shape


# In[39]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[40]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Create and train the ANN Classifier

# In[41]:


from keras.models import Sequential
from keras.layers import Dense


# In[62]:


clf_ann = Sequential()

# First Hidden Layer
clf_ann.add(Dense(output_dim = 84, init = 'uniform', activation = 'relu', input_dim = 42))

# Second Hidden Layer
clf_ann.add(Dense(output_dim = 84, init = 'uniform', activation = 'relu'))

#clf_ann.add(Dense(output_dim = 21, init = 'uniform', activation = 'relu'))

# Output Layer
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN on the Train Data
clf_ann.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# In[63]:


# Test the ANN on the Test Data
Y_pred = clf_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)


# ## Check the Accuracy

# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[65]:


accuracy_score(Y_pred, Y_test)


# In[66]:


confusion_matrix(Y_pred, Y_test)


# In[ ]:




