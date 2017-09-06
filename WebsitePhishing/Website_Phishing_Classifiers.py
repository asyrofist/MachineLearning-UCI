
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[3]:


model_accuracies = {'LogReg':1, 'DT':1, 'RF':1, 'LinearSVC':1, 'KernelSVC':1, 'NB':1, 'KNN':1}


# ## Importing the data

# In[4]:


dataset = pd.read_csv('PhishingData.csv', header = None)
dataset.shape


# In[5]:


dataset.head()


# ## Create X and Y

# In[6]:


X = dataset.iloc[:, 0:9].values
Y = dataset.iloc[:, 9].values


# In[7]:


X.shape


# In[8]:


Y.shape


# In[9]:


X[0]


# In[10]:


Y


# ## Preprocess the Data

# In[11]:


le_Y = LabelEncoder()


# In[12]:


Y = le_Y.fit_transform(Y)
Y


# In[13]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[14]:


for x in range(0, 9):
    encoder_X(x)


# In[15]:


X[0]


# In[16]:


pd.DataFrame(pd.DataFrame(X[:, 6])[0].value_counts())


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[18]:


X = ohe_X.fit_transform(X).toarray()
X[0]


# In[19]:


X.shape


# In[20]:


X = X[:, 1:]
X.shape


# In[21]:


pd.DataFrame(pd.DataFrame(X[:, 7])[0].value_counts())


# In[22]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[23]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[24]:


X = X[:, 1:]
X.shape


# In[25]:


pd.DataFrame(pd.DataFrame(X[:, 8])[0].value_counts())


# In[26]:


ohe_X = OneHotEncoder(categorical_features = [8])


# In[27]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


pd.DataFrame(pd.DataFrame(X[:, 9])[0].value_counts())


# In[29]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[30]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[31]:


pd.DataFrame(pd.DataFrame(X[:, 10])[0].value_counts())


# In[32]:


ohe_X = OneHotEncoder(categorical_features = [10])


# In[33]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[34]:


pd.DataFrame(pd.DataFrame(X[:, 11])[0].value_counts())


# In[35]:


ohe_X = OneHotEncoder(categorical_features = [11])


# In[36]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[37]:


pd.DataFrame(pd.DataFrame(X[:, 12])[0].value_counts())


# In[38]:


ohe_X = OneHotEncoder(categorical_features = [12])


# In[39]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# ## Create Train and Test Data

# In[40]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[41]:


X_train.shape


# In[42]:


X_test.shape


# In[43]:


Y_train.shape


# In[44]:


Y_test.shape


# In[45]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[46]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[47]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[48]:


clf_dt.fit(X_train, Y_train)


# In[49]:


Y_pred_dt = clf_dt.predict(X_test)


# In[50]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest

# In[51]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[52]:


clf_rf.fit(X_train, Y_train)


# In[53]:


Y_pred_rf = clf_rf.predict(X_test)


# In[54]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes

# In[55]:


clf_nb = GaussianNB()


# In[56]:


clf_nb.fit(X_train, Y_train)


# In[57]:


Y_pred_nb = clf_nb.predict(X_test)


# In[58]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN

# In[59]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[60]:


clf_knn.fit(X_train, Y_train)


# In[61]:


Y_pred_knn = clf_knn.predict(X_test)


# In[62]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[63]:


clf_lr = LogisticRegression()


# In[64]:


clf_lr.fit(X_train, Y_train)


# In[65]:


Y_pred_lr = clf_lr.predict(X_test)


# In[66]:


confusion_matrix(Y_test, Y_pred_lr)


# ## Linear SVC

# In[67]:


clf_lsvc = SVC(kernel = 'linear')


# In[68]:


clf_lsvc.fit(X_train, Y_train)


# In[69]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[70]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## Kernel SVC

# In[71]:


clf_ksvc = SVC(kernel = 'rbf')


# In[72]:


clf_ksvc.fit(X_train, Y_train)


# In[73]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[74]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Models

# In[75]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies


# In[ ]:





# In[ ]:




