
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


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


df = pd.read_csv('transfusion.data')
df.shape


# In[6]:


df.head()


# ## Create X and Y

# In[7]:


Y = df.iloc[:, 4].values
Y.shape


# In[8]:


X = df.iloc[:, 0:4].values
X.shape


# ## Preprocess the Data

# In[9]:


sc_X = StandardScaler()


# In[10]:


X = sc_X.fit_transform(X)


# In[11]:


X[0]


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


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest Classifier

# In[23]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[24]:


clf_rf.fit(X_train, Y_train)


# In[25]:


Y_pred_rf = clf_rf.predict(X_test)


# In[26]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes Classifier

# In[27]:


clf_nb = GaussianNB()


# In[28]:


clf_nb.fit(X_train, Y_train)


# In[29]:


Y_pred_nb = clf_nb.predict(X_test)


# In[30]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN Classifier

# In[31]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[32]:


clf_knn.fit(X_train, Y_train)


# In[33]:


Y_pred_knn = clf_knn.predict(X_test)


# In[34]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[35]:


clf_lr = LogisticRegression()


# In[36]:


clf_lr.fit(X_train, Y_train)


# In[37]:


Y_pred_lr = clf_lr.predict(X_test)


# In[38]:


confusion_matrix(Y_test, Y_pred_lr)


# ## SVC Linear

# In[39]:


clf_lsvc = SVC(kernel = "linear")


# In[40]:


clf_lsvc.fit(X_train, Y_train)


# In[41]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[42]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## SVC Kernel

# In[43]:


clf_ksvc = SVC(kernel = "rbf")


# In[44]:


clf_ksvc.fit(X_train, Y_train)


# In[45]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[46]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Check the Accuracy of Various Classifiers

# In[47]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies


# In[ ]:




