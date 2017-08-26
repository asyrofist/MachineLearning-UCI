
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


dataset = pd.read_csv('ThoraricSurgery.csv', header = None)
dataset.shape


# In[5]:


dataset.head()


# ## Create X and Y

# In[6]:


X = dataset.iloc[:, 0:16].values
Y = dataset.iloc[:, 16].values


# In[7]:


X.shape


# In[8]:


Y.shape


# In[9]:


X


# In[10]:


Y


# ## Preprocess the Data

# In[11]:


le_Y = LabelEncoder()


# In[12]:


Y = le_Y.fit_transform(Y)
Y


# In[13]:


def enocder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[14]:


X_indices = [0] + list(range(3,15))
X_indices


# In[15]:


for x in X_indices:
    enocder_X(x)


# In[16]:


X


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[18]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[19]:


X = X[:, 1:]
X.shape


# In[20]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[21]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[22]:


X = X[:, 1:]
X.shape


# In[23]:


ohe_X = OneHotEncoder(categorical_features = [5])


# In[24]:


X = ohe_X.fit_transform(X).toarray()
X.shape


# In[25]:


X = X[:, 1:]
X.shape


# In[26]:


sc_X = StandardScaler()


# In[27]:


X = sc_X.fit_transform(X)
X


# ## Create Train and Test Data

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


# ## DecisionTree

# In[35]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[36]:


clf_dt.fit(X_train, Y_train)


# In[37]:


Y_pred_dt = clf_dt.predict(X_test)


# In[38]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[39]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[40]:


clf_rf.fit(X_train, Y_train)


# In[41]:


Y_pred_rf = clf_rf.predict(X_test)


# In[42]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[43]:


clf_nb = GaussianNB()


# In[44]:


clf_nb.fit(X_train, Y_train)


# In[45]:


Y_pred_nb = clf_nb.predict(X_test)


# In[46]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[47]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[48]:


clf_knn.fit(X_train, Y_train)


# In[49]:


Y_pred_knn = clf_knn.predict(X_test)


# In[50]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[51]:


clf_lr = LogisticRegression()


# In[52]:


clf_lr.fit(X_train, Y_train)


# In[53]:


Y_pred_lr = clf_lr.predict(X_test)


# In[54]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[55]:


clf_lsvc = SVC(kernel = 'linear')


# In[56]:


clf_lsvc.fit(X_train, Y_train)


# In[57]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[58]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[59]:


clf_ksvc = SVC(kernel = 'rbf')


# In[60]:


clf_ksvc.fit(X_train, Y_train)


# In[61]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[62]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[63]:


model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)
model_accuracies


# In[ ]:





# In[ ]:




