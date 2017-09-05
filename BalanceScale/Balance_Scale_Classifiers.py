
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd


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


model_accuracies = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# ## Importing the Data

# In[5]:


df = pd.read_csv('balance-scale.data', header = None)
df.shape


# In[6]:


df.head()


# ## Creating X and Y

# In[7]:


X = df.iloc[:, 1:5].values
Y = df.iloc[:, 0].values


# In[8]:


X


# In[9]:


Y


# ## Preprocess the Data

# In[10]:


le_Y = LabelEncoder()


# In[11]:


Y = le_Y.fit_transform(Y)
Y


# ## Create Train and Test data

# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


Y_train.shape


# In[16]:


Y_test.shape


# In[17]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[18]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[19]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[20]:


clf_dt.fit(X_train, Y_train)


# In[21]:


Y_pred_dt = clf_dt.predict(X_test)


# In[22]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[23]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[24]:


clf_rf.fit(X_train, Y_train)


# In[25]:


Y_pred_rf = clf_rf.predict(X_test)


# In[26]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[27]:


clf_nb = GaussianNB()


# In[28]:


clf_nb.fit(X_train, Y_train)


# In[29]:


Y_pred_nb = clf_nb.predict(X_test)


# In[30]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[31]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[32]:


clf_knn.fit(X_train, Y_train)


# In[33]:


Y_pred_knn = clf_knn.predict(X_test)


# In[34]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[35]:


clf_lr = LogisticRegression()


# In[36]:


clf_lr.fit(X_train, Y_train)


# In[37]:


Y_pred_lr = clf_lr.predict(X_test)


# In[38]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[39]:


clf_lsvc = SVC(kernel = "linear")


# In[40]:


clf_lsvc.fit(X_train, Y_train)


# In[41]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[42]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[43]:


clf_ksvc = SVC(kernel = "rbf")


# In[44]:


clf_ksvc.fit(X_train, Y_train)


# In[45]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[46]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[47]:


model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)
model_accuracies

