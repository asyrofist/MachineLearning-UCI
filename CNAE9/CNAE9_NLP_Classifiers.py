
# coding: utf-8

# ## Initialization

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[6]:


model_accuracies = {'LogReg':1, 'DT':1, 'RF':1, 'LinearSVC':1, 'KernelSVC':1, 'NB':1, 'KNN':1}


# ## Importing the data

# In[7]:


dataset = pd.read_csv('CNAE-9.csv', header = None)
dataset.shape


# In[8]:


dataset.head()


# ## Create X and Y

# In[9]:


X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, 0]


# ## Create Train and Test Data

# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


Y_train.shape


# In[14]:


Y_test.shape


# In[15]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[16]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[17]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[18]:


clf_dt.fit(X_train, Y_train)


# In[19]:


Y_pred_dt = clf_dt.predict(X_test)


# In[20]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[21]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[22]:


clf_rf.fit(X_train, Y_train)


# In[23]:


Y_pred_rf = clf_rf.predict(X_test)


# In[24]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[25]:


clf_nb = GaussianNB()


# In[26]:


clf_nb.fit(X_train, Y_train)


# In[27]:


Y_pred_nb = clf_nb.predict(X_test)


# In[28]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[29]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[30]:


clf_knn.fit(X_train, Y_train)


# In[31]:


Y_pred_knn = clf_knn.predict(X_test)


# In[32]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[33]:


clf_lr = LogisticRegression()


# In[34]:


clf_lr.fit(X_train, Y_train)


# In[35]:


Y_pred_lr = clf_lr.predict(X_test)


# In[36]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[37]:


clf_lsvc = SVC(kernel = 'linear')


# In[38]:


clf_lsvc.fit(X_train, Y_train)


# In[39]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[40]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[41]:


clf_ksvc = SVC(kernel = 'rbf')


# In[42]:


clf_ksvc.fit(X_train, Y_train)


# In[43]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[44]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[45]:


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




