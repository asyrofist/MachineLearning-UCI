
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[32]:


model_accuracies_1 = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}
model_accuracies_2 = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# ## Importing the Data

# In[5]:


dataset = pd.read_csv("diagnosis.data", header = None, delimiter = r"\s+")
X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values


# In[6]:


dataset.head()


# ## Preprocess the Data

# In[8]:


le_Y = LabelEncoder()
Y1 = le_Y.fit_transform(Y1)
Y2 = le_Y.transform(Y2)


# In[9]:


Y1


# In[10]:


Y2


# In[11]:


le_X = LabelEncoder()
le_X.fit(X[:, 1])

X[:, 1] = le_X.transform(X[:, 1])
X[:, 2] = le_X.transform(X[:, 2])
X[:, 3] = le_X.transform(X[:, 3])
X[:, 4] = le_X.transform(X[:, 4])
X[:, 5] = le_X.transform(X[:, 5])


# In[12]:


X


# ## Create Train and Test Data

# In[13]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 4)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 4)


# ## Decision Tree Classifier Y1

# In[14]:


clf_dt_1 = DecisionTreeClassifier(criterion = 'entropy')


# In[15]:


clf_dt_1.fit(X1_train, Y1_train)


# In[16]:


Y1_pred_dt = clf_dt_1.predict(X1_test)


# In[17]:


cm1_dt = confusion_matrix(Y1_pred_dt, Y1_test)
cm1_dt


# ## Decision Tree Classifier Y2

# In[18]:


clf_dt_2 = DecisionTreeClassifier(criterion = 'entropy')


# In[19]:


clf_dt_2.fit(X2_train, Y2_train)


# In[20]:


Y2_pred_dt = clf_dt_2.predict(X2_test)


# In[21]:


cm2_dt = confusion_matrix(Y2_pred_dt, Y2_test)
cm2_dt


# ## Random Forest Classifier Y1

# In[22]:


clf_rf_1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[23]:


clf_rf_1.fit(X1_train, Y1_train)


# In[25]:


Y1_pred_rf = clf_rf_1.predict(X1_test)


# In[26]:


cm1_rf = confusion_matrix(Y1_pred_dt, Y1_test)
cm1_rf


# ## Random Forest Classifier Y2

# In[27]:


clf_rf_2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[28]:


clf_rf_2.fit(X2_train, Y2_train)


# In[29]:


Y2_pred_rf = clf_rf_2.predict(X2_test)


# In[30]:


cm2_rf = confusion_matrix(Y2_pred_dt, Y2_test)
cm2_rf


# ## Naive Bayes Classifier Y1

# In[33]:


clf_nb_1 = GaussianNB()


# In[34]:


clf_nb_1.fit(X1_train, Y1_train)


# In[35]:


Y1_pred_nb = clf_nb_1.predict(X1_test)


# In[36]:


cm1_nb = confusion_matrix(Y1_pred_nb, Y1_test)
cm1_nb


# ## Naive Bayes Classifier Y2

# In[37]:


clf_nb_2 = GaussianNB()


# In[38]:


clf_nb_2.fit(X2_train, Y2_train)


# In[39]:


Y2_pred_nb = clf_nb_2.predict(X2_test)


# In[40]:


cm2_nb = confusion_matrix(Y2_pred_nb, Y2_test)
cm2_nb


# ## KNN Classifier Y1

# In[41]:


clf_knn_1 = KNeighborsClassifier(n_neighbors = 5)


# In[42]:


clf_knn_1.fit(X1_train, Y1_train)


# In[43]:


Y1_pred_knn = clf_knn_1.predict(X1_test)


# In[44]:


cm1_knn = confusion_matrix(Y1_pred_knn, Y1_test)
cm1_knn


# ## KNN Classifier Y2

# In[45]:


clf_knn_2 = KNeighborsClassifier(n_neighbors = 5)


# In[46]:


clf_knn_2.fit(X2_train, Y2_train)


# In[47]:


Y2_pred_knn = clf_knn_2.predict(X2_test)


# In[48]:


cm2_knn = confusion_matrix(Y2_pred_knn, Y2_test)
cm2_knn


# ## Logistic Regression Classifier Y1

# In[49]:


clf_lr_1 = LogisticRegression()


# In[50]:


clf_lr_1.fit(X1_train, Y1_train)


# In[51]:


Y1_pred_lr = clf_lr_1.predict(X1_test)


# In[52]:


cm1_lr = confusion_matrix(Y1_pred_lr, Y1_test)
cm1_lr


# ## Logistic Regression Classifier Y2

# In[53]:


clf_lr_2 = LogisticRegression()


# In[54]:


clf_lr_2.fit(X2_train, Y2_train)


# In[55]:


Y2_pred_lr = clf_lr_2.predict(X2_test)


# In[56]:


cm2_lr = confusion_matrix(Y2_pred_lr, Y2_test)
cm2_lr


# ## Linear SVC Classifier Y1

# In[57]:


clf_lsvc_1 = SVC(kernel = 'linear')


# In[58]:


clf_lsvc_1.fit(X1_train, Y1_train)


# In[59]:


Y1_pred_lsvc = clf_lsvc_1.predict(X1_test)


# In[61]:


cm1_lsvc = confusion_matrix(Y1_pred_lsvc, Y1_test)
cm1_lsvc


# ## Linear SVC Classifier Y2

# In[63]:


clf_lsvc_2 = SVC(kernel = 'linear')


# In[64]:


clf_lsvc_2.fit(X2_train, Y2_train)


# In[65]:


Y2_pred_lsvc = clf_lsvc_2.predict(X2_test)


# In[66]:


cm2_lsvc = confusion_matrix(Y2_pred_lsvc, Y2_test)
cm2_lsvc


# ## Kernel SVC Classifier Y1

# In[67]:


clf_ksvc_1 = SVC(kernel = 'rbf')


# In[68]:


clf_ksvc_1.fit(X1_train, Y1_train)


# In[69]:


Y1_pred_ksvc = clf_ksvc_1.predict(X1_test)


# In[70]:


cm1_ksvc = confusion_matrix(Y1_pred_ksvc, Y1_test)
cm1_ksvc


# ## Kernel SVC Classifier Y2

# In[71]:


clf_ksvc_2 = SVC(kernel = 'rbf')


# In[72]:


clf_ksvc_2.fit(X2_train, Y2_train)


# In[73]:


Y2_pred_ksvc = clf_ksvc_2.predict(X2_test)


# In[74]:


cm2_ksvc = confusion_matrix(Y2_pred_ksvc, Y2_test)
cm2_ksvc


# ## Checking the Model Accuracies

# In[75]:


model_accuracies_1['DT'] = accuracy_score(Y1_pred_dt, Y1_test)
model_accuracies_1['KNN'] = accuracy_score(Y1_pred_knn, Y1_test)
model_accuracies_1['KernelSVC'] = accuracy_score(Y1_pred_ksvc, Y1_test)
model_accuracies_1['LinearSVC'] = accuracy_score(Y1_pred_lsvc, Y1_test)
model_accuracies_1['LogReg'] = accuracy_score(Y1_pred_lr, Y1_test)
model_accuracies_1['NB'] = accuracy_score(Y1_pred_nb, Y1_test)
model_accuracies_1['RF'] = accuracy_score(Y1_pred_rf, Y1_test)
model_accuracies_1


# In[76]:


model_accuracies_2['DT'] = accuracy_score(Y2_pred_dt, Y2_test)
model_accuracies_2['KNN'] = accuracy_score(Y2_pred_knn, Y2_test)
model_accuracies_2['KernelSVC'] = accuracy_score(Y2_pred_ksvc, Y2_test)
model_accuracies_2['LinearSVC'] = accuracy_score(Y2_pred_lsvc, Y2_test)
model_accuracies_2['LogReg'] = accuracy_score(Y2_pred_lr, Y2_test)
model_accuracies_2['NB'] = accuracy_score(Y2_pred_nb, Y2_test)
model_accuracies_2['RF'] = accuracy_score(Y2_pred_rf, Y2_test)
model_accuracies_2


# In[ ]:




