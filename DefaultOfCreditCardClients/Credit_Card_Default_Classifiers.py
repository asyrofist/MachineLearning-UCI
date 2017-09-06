
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[4]:


model_accuracies = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# ## Importing the Data

# In[5]:


dataset = pd.read_excel('Dataset.xls')
dataset.shape


# In[6]:


dataset.head()


# ## Create X and Y

# In[7]:


Y = dataset.iloc[:, 23].values
Y.shape


# In[8]:


Y[0]


# In[9]:


X = dataset.iloc[:, 0:23].values
X.shape


# In[10]:


X[0]


# ##  Preprocess the Data

# In[11]:


le_X = LabelEncoder()
X[:, 1] = le_X.fit_transform(X[:, 1])


# In[12]:


le_X = LabelEncoder()
X[:, 2] = le_X.fit_transform(X[:, 2])


# In[13]:


le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])


# In[14]:


ohe_X = OneHotEncoder(categorical_features = [1])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[15]:


ohe_X = OneHotEncoder(categorical_features = [2])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[16]:


ohe_X = OneHotEncoder(categorical_features = [8])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[17]:


sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[18]:


pd.DataFrame(X).head()


# ## Create Train and Test data

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


Y_train.shape


# In[23]:


Y_test.shape


# In[24]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[25]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[26]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[27]:


clf_dt.fit(X_train, Y_train)


# In[28]:


Y_pred_dt = clf_dt.predict(X_test)


# In[29]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest Classifier

# In[30]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[31]:


clf_rf.fit(X_train, Y_train)


# In[32]:


Y_pred_rf = clf_rf.predict(X_test)


# In[33]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes Classifier

# In[34]:


clf_nb = GaussianNB()


# In[35]:


clf_nb.fit(X_train, Y_train)


# In[36]:


Y_pred_nb = clf_nb.predict(X_test)


# In[37]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN Classifier

# In[38]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[39]:


clf_knn.fit(X_train, Y_train)


# In[40]:


Y_pred_knn = clf_knn.predict(X_test)


# In[41]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[42]:


clf_lr = LogisticRegression()


# In[43]:


clf_lr.fit(X_train, Y_train)


# In[44]:


Y_pred_lr = clf_lr.predict(X_test)


# In[45]:


confusion_matrix(Y_test, Y_pred_lr)


# ## SVC Linear

# In[46]:


clf_lsvc = SVC(kernel = "linear")


# In[47]:


clf_lsvc.fit(X_train, Y_train)


# In[48]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[49]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# In[50]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## SVC Kernel

# In[51]:


clf_ksvc = SVC(kernel = "rbf")


# In[52]:


clf_ksvc.fit(X_train, Y_train)


# In[53]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[54]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Classifiers

# In[55]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies


# In[ ]:




