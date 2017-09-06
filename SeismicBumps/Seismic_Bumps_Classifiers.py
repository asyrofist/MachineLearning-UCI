
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


dataset = pd.read_csv('Seismic Bumps.csv', header = None)
dataset.shape


# In[5]:


dataset.head()


# In[6]:


pd.DataFrame(pd.DataFrame(dataset.iloc[:, 7])[7].value_counts())


# In[7]:


pd.DataFrame(pd.DataFrame(dataset.iloc[:, 2])[2].value_counts())


# In[8]:


pd.DataFrame(pd.DataFrame(dataset.iloc[:, 1])[1].value_counts())


# In[9]:


pd.DataFrame(pd.DataFrame(dataset.iloc[:, 0])[0].value_counts())


# ## Create X and Y

# In[10]:


X = dataset.iloc[:, 0:18].values
Y = dataset.iloc[:, 18].values


# In[11]:


X.shape


# In[12]:


Y.shape


# In[13]:


X[0]


# In[14]:


Y[0]


# ## Preprocess the Data

# In[15]:


le_Y = LabelEncoder()


# In[16]:


Y = le_Y.fit_transform(Y)


# In[17]:


Y[0]


# In[18]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[19]:


to_be_encoded_indices = [0, 1, 2, 7]


# In[20]:


for x in to_be_encoded_indices:
    encoder_X(x)


# In[21]:


X[0]


# In[22]:


pd.DataFrame(pd.DataFrame(X[:, 7])[0].value_counts())


# In[23]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[24]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[25]:


pd.DataFrame(pd.DataFrame(X[:, 3])[0].value_counts())


# In[26]:


ohe_X = OneHotEncoder(categorical_features = [3])


# In[27]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


sc_X = StandardScaler()


# In[29]:


X = sc_X.fit_transform(X)


# In[30]:


X[0]


# ## Create Train and Test Data

# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[32]:


X_train.shape


# In[33]:


X_test.shape


# In[34]:


Y_train.shape


# In[35]:


Y_test.shape


# In[36]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[37]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[38]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[39]:


clf_dt.fit(X_train, Y_train)


# In[40]:


Y_pred_dt = clf_dt.predict(X_test)


# In[41]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest

# In[42]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[43]:


clf_rf.fit(X_train, Y_train)


# In[44]:


Y_pred_rf = clf_rf.predict(X_test)


# In[45]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes

# In[46]:


clf_nb = GaussianNB()


# In[47]:


clf_nb.fit(X_train, Y_train)


# In[48]:


Y_pred_nb = clf_nb.predict(X_test)


# In[49]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN

# In[50]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[51]:


clf_knn.fit(X_train, Y_train)


# In[52]:


Y_pred_knn = clf_knn.predict(X_test)


# In[53]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[54]:


clf_lr = LogisticRegression()


# In[55]:


clf_lr.fit(X_train, Y_train)


# In[56]:


Y_pred_lr = clf_lr.predict(X_test)


# In[57]:


confusion_matrix(Y_test, Y_pred_lr)


# ## Linear SVC

# In[58]:


clf_lsvc = SVC(kernel = 'linear')


# In[59]:


clf_lsvc.fit(X_train, Y_train)


# In[60]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[61]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## Kernel SVC

# In[62]:


clf_ksvc = SVC(kernel = 'rbf')


# In[63]:


clf_ksvc.fit(X_train, Y_train)


# In[64]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[65]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Models

# In[66]:


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




