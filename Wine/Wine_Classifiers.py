
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


df = pd.read_csv('wine.data', header = None)
df.shape


# In[6]:


df.head()


# ## Creating X and Y

# In[7]:


Y = df.iloc[:, 0]
Y.shape


# In[8]:


Y


# In[9]:


X = df.iloc[:, 1:]
X.shape


# In[10]:


X


# ## Preprocess the Data

# In[11]:


le_Y = LabelEncoder()


# In[12]:


Y = le_Y.fit_transform(Y)
Y


# In[13]:


sc_X = StandardScaler()


# In[14]:


X = sc_X.fit_transform(X)
X


# ## Create Train and Test data

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


Y_train.shape


# In[19]:


Y_test.shape


# In[20]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[21]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[22]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[23]:


clf_dt.fit(X_train, Y_train)


# In[24]:


Y_pred_dt = clf_dt.predict(X_test)


# In[25]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[26]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[27]:


clf_rf.fit(X_train, Y_train)


# In[28]:


Y_pred_rf = clf_rf.predict(X_test)


# In[29]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[30]:


clf_nb = GaussianNB()


# In[31]:


clf_nb.fit(X_train, Y_train)


# In[32]:


Y_pred_nb = clf_nb.predict(X_test)


# In[33]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[34]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[35]:


clf_knn.fit(X_train, Y_train)


# In[36]:


Y_pred_knn = clf_knn.predict(X_test)


# In[37]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[38]:


clf_lr = LogisticRegression()


# In[39]:


clf_lr.fit(X_train, Y_train)


# In[40]:


Y_pred_lr = clf_lr.predict(X_test)


# In[41]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[42]:


clf_lsvc = SVC(kernel = "linear")


# In[43]:


clf_lsvc.fit(X_train, Y_train)


# In[44]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[45]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[46]:


clf_ksvc = SVC(kernel = "rbf")


# In[47]:


clf_ksvc.fit(X_train, Y_train)


# In[48]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[49]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[50]:


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




