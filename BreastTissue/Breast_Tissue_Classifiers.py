
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


df = pd.read_excel('BreastTissue.xls', sheetname = 1)
df.shape


# In[6]:


df.head()


# ## Create X and Y

# In[7]:


Y = df.iloc[:, 1].values
Y.shape


# In[8]:


Y[0]


# In[9]:


X = df.iloc[:, 2:].values
X.shape


# In[10]:


X[0]


# ## Preprocess the Data

# In[11]:


sc_X = StandardScaler()


# In[12]:


X = sc_X.fit_transform(X)


# In[13]:


le_Y = LabelEncoder()


# In[14]:


Y = le_Y.fit_transform(Y)


# In[15]:


X[0]


# In[16]:


Y[0]


# ## Create Train and Test data

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


Y_train.shape


# In[21]:


Y_test.shape


# In[22]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[23]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[24]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[25]:


clf_dt.fit(X_train, Y_train)


# In[26]:


Y_pred_dt = clf_dt.predict(X_test)


# In[27]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest Classifier

# In[28]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[29]:


clf_rf.fit(X_train, Y_train)


# In[30]:


Y_pred_rf = clf_rf.predict(X_test)


# In[31]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes Classifier

# In[32]:


clf_nb = GaussianNB()


# In[33]:


clf_nb.fit(X_train, Y_train)


# In[34]:


Y_pred_nb = clf_nb.predict(X_test)


# In[35]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN Classifier

# In[36]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[37]:


clf_knn.fit(X_train, Y_train)


# In[38]:


Y_pred_knn = clf_knn.predict(X_test)


# In[39]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[40]:


clf_lr = LogisticRegression()


# In[41]:


clf_lr.fit(X_train, Y_train)


# In[42]:


Y_pred_lr = clf_lr.predict(X_test)


# In[43]:


confusion_matrix(Y_test, Y_pred_lr)


# ## SVC Linear

# In[44]:


clf_lsvc = SVC(kernel = "linear")


# In[45]:


clf_lsvc.fit(X_train, Y_train)


# In[46]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[47]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## SVC Kernel

# In[48]:


clf_ksvc = SVC(kernel = "rbf")


# In[49]:


clf_ksvc.fit(X_train, Y_train)


# In[50]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[51]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Classifiers

# In[52]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies


# In[ ]:




