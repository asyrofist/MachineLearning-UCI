
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

# In[19]:


dataset = pd.read_csv('Dataset.csv', delimiter=';')
dataset.shape


# In[20]:


dataset.head()


# ## Create X and Y

# In[21]:


X = dataset.iloc[:, 1:18].values
Y = dataset.iloc[:, 18].values


# In[22]:


X.shape


# In[23]:


Y.shape


# In[24]:


X


# In[25]:


Y


# ## Preprocess the Data

# In[26]:


le_Y = LabelEncoder()


# In[27]:


Y = le_Y.fit_transform(Y)


# In[28]:


Y


# In[29]:


le_X = LabelEncoder()


# In[30]:


X[:, 0] = le_X.fit_transform(X[:, 0])


# In[31]:


X


# In[32]:


sc_X = StandardScaler()


# In[33]:


X = sc_X.fit_transform(X)


# In[34]:


X


# ## Create Train and Test Data

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


Y_train.shape


# In[39]:


Y_test.shape


# In[40]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[41]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[42]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[43]:


clf_dt.fit(X_train, Y_train)


# In[44]:


Y_pred_dt = clf_dt.predict(X_test)


# In[45]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[71]:


clf_rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')


# In[72]:


clf_rf.fit(X_train, Y_train)


# In[73]:


Y_pred_rf = clf_rf.predict(X_test)


# In[74]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[50]:


clf_nb = GaussianNB()


# In[51]:


clf_nb.fit(X_train, Y_train)


# In[52]:


Y_pred_nb = clf_nb.predict(X_test)


# In[53]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[54]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[55]:


clf_knn.fit(X_train, Y_train)


# In[56]:


Y_pred_knn = clf_knn.predict(X_test)


# In[57]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[58]:


clf_lr = LogisticRegression()


# In[59]:


clf_lr.fit(X_train, Y_train)


# In[60]:


Y_pred_lr = clf_lr.predict(X_test)


# In[61]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[62]:


clf_lsvc = SVC(kernel = 'linear')


# In[63]:


clf_lsvc.fit(X_train, Y_train)


# In[64]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[65]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[66]:


clf_ksvc = SVC(kernel = 'rbf')


# In[67]:


clf_ksvc.fit(X_train, Y_train)


# In[68]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[69]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[75]:


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




