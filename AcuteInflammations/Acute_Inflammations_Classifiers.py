
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


# In[4]:


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

# In[7]:


le_Y = LabelEncoder()
Y1 = le_Y.fit_transform(Y1)
Y2 = le_Y.transform(Y2)


# In[8]:


Y1


# In[9]:


Y2


# In[10]:


le_X = LabelEncoder()
le_X.fit(X[:, 1])

X[:, 1] = le_X.transform(X[:, 1])
X[:, 2] = le_X.transform(X[:, 2])
X[:, 3] = le_X.transform(X[:, 3])
X[:, 4] = le_X.transform(X[:, 4])
X[:, 5] = le_X.transform(X[:, 5])


# In[11]:


X[0]


# ## Create Train and Test Data

# In[12]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 4)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 4)


# ## Decision Tree Classifier Y1

# In[13]:


clf_dt_1 = DecisionTreeClassifier(criterion = 'entropy')


# In[14]:


clf_dt_1.fit(X1_train, Y1_train)


# In[15]:


Y1_pred_dt = clf_dt_1.predict(X1_test)


# In[16]:


confusion_matrix(Y1_test, Y1_pred_dt)


# ## Decision Tree Classifier Y2

# In[17]:


clf_dt_2 = DecisionTreeClassifier(criterion = 'entropy')


# In[18]:


clf_dt_2.fit(X2_train, Y2_train)


# In[19]:


Y2_pred_dt = clf_dt_2.predict(X2_test)


# In[20]:


confusion_matrix(Y2_test, Y2_pred_dt)


# ## Random Forest Classifier Y1

# In[21]:


clf_rf_1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[22]:


clf_rf_1.fit(X1_train, Y1_train)


# In[23]:


Y1_pred_rf = clf_rf_1.predict(X1_test)


# In[24]:


confusion_matrix(Y1_test, Y1_pred_dt)


# ## Random Forest Classifier Y2

# In[25]:


clf_rf_2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[26]:


clf_rf_2.fit(X2_train, Y2_train)


# In[27]:


Y2_pred_rf = clf_rf_2.predict(X2_test)


# In[28]:


confusion_matrix(Y2_test, Y2_pred_dt)


# ## Naive Bayes Classifier Y1

# In[29]:


clf_nb_1 = GaussianNB()


# In[30]:


clf_nb_1.fit(X1_train, Y1_train)


# In[31]:


Y1_pred_nb = clf_nb_1.predict(X1_test)


# In[32]:


confusion_matrix(Y1_test, Y1_pred_nb)


# ## Naive Bayes Classifier Y2

# In[33]:


clf_nb_2 = GaussianNB()


# In[34]:


clf_nb_2.fit(X2_train, Y2_train)


# In[35]:


Y2_pred_nb = clf_nb_2.predict(X2_test)


# In[36]:


confusion_matrix(Y2_test, Y2_pred_nb)


# ## KNN Classifier Y1

# In[37]:


clf_knn_1 = KNeighborsClassifier(n_neighbors = 5)


# In[38]:


clf_knn_1.fit(X1_train, Y1_train)


# In[39]:


Y1_pred_knn = clf_knn_1.predict(X1_test)


# In[40]:


confusion_matrix(Y1_test, Y1_pred_knn)


# ## KNN Classifier Y2

# In[41]:


clf_knn_2 = KNeighborsClassifier(n_neighbors = 5)


# In[42]:


clf_knn_2.fit(X2_train, Y2_train)


# In[43]:


Y2_pred_knn = clf_knn_2.predict(X2_test)


# In[44]:


confusion_matrix(Y2_test, Y2_pred_knn)


# ## Logistic Regression Classifier Y1

# In[45]:


clf_lr_1 = LogisticRegression()


# In[46]:


clf_lr_1.fit(X1_train, Y1_train)


# In[47]:


Y1_pred_lr = clf_lr_1.predict(X1_test)


# In[48]:


confusion_matrix(Y1_test, Y1_pred_lr)


# ## Logistic Regression Classifier Y2

# In[49]:


clf_lr_2 = LogisticRegression()


# In[50]:


clf_lr_2.fit(X2_train, Y2_train)


# In[51]:


Y2_pred_lr = clf_lr_2.predict(X2_test)


# In[52]:


confusion_matrix(Y2_test, Y2_pred_lr)


# ## Linear SVC Classifier Y1

# In[53]:


clf_lsvc_1 = SVC(kernel = 'linear')


# In[54]:


clf_lsvc_1.fit(X1_train, Y1_train)


# In[55]:


Y1_pred_lsvc = clf_lsvc_1.predict(X1_test)


# In[56]:


confusion_matrix(Y1_test, Y1_pred_lsvc)


# ## Linear SVC Classifier Y2

# In[57]:


clf_lsvc_2 = SVC(kernel = 'linear')


# In[58]:


clf_lsvc_2.fit(X2_train, Y2_train)


# In[59]:


Y2_pred_lsvc = clf_lsvc_2.predict(X2_test)


# In[60]:


confusion_matrix(Y2_test, Y2_pred_lsvc)


# ## Kernel SVC Classifier Y1

# In[61]:


clf_ksvc_1 = SVC(kernel = 'rbf')


# In[62]:


clf_ksvc_1.fit(X1_train, Y1_train)


# In[63]:


Y1_pred_ksvc = clf_ksvc_1.predict(X1_test)


# In[64]:


confusion_matrix(Y1_test, Y1_pred_ksvc)


# ## Kernel SVC Classifier Y2

# In[65]:


clf_ksvc_2 = SVC(kernel = 'rbf')


# In[66]:


clf_ksvc_2.fit(X2_train, Y2_train)


# In[67]:


Y2_pred_ksvc = clf_ksvc_2.predict(X2_test)


# In[68]:


confusion_matrix(Y2_test, Y2_pred_ksvc)


# ## Checking the Model Accuracies

# In[69]:


model_accuracies_1['DT'] = accuracy_score(Y1_test, Y1_pred_dt)
model_accuracies_1['KNN'] = accuracy_score(Y1_test, Y1_pred_knn)
model_accuracies_1['KernelSVC'] = accuracy_score(Y1_test, Y1_pred_ksvc)
model_accuracies_1['LinearSVC'] = accuracy_score(Y1_test, Y1_pred_lsvc)
model_accuracies_1['LogReg'] = accuracy_score(Y1_test, Y1_pred_lr)
model_accuracies_1['NB'] = accuracy_score(Y1_test, Y1_pred_nb)
model_accuracies_1['RF'] = accuracy_score(Y1_test, Y1_pred_rf)
model_accuracies_1


# In[70]:


model_accuracies_2['DT'] = accuracy_score(Y2_test, Y2_pred_dt)
model_accuracies_2['KNN'] = accuracy_score(Y2_test, Y2_pred_knn)
model_accuracies_2['KernelSVC'] = accuracy_score(Y2_test, Y2_pred_ksvc)
model_accuracies_2['LinearSVC'] = accuracy_score(Y2_test, Y2_pred_lsvc)
model_accuracies_2['LogReg'] = accuracy_score(Y2_test, Y2_pred_lr)
model_accuracies_2['NB'] = accuracy_score(Y2_test, Y2_pred_nb)
model_accuracies_2['RF'] = accuracy_score(Y2_test, Y2_pred_rf)
model_accuracies_2


# In[ ]:




