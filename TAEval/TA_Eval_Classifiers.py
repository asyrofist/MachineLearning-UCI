
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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


df = pd.read_csv('tae.data', header = None)
df.shape


# In[6]:


df.head()


# ## Creating X and Y

# In[7]:


X = df.iloc[:, 0:5].values
Y = df.iloc[:, 5].values


# In[8]:


Y.shape


# In[9]:


X.shape


# In[10]:


X.shape[1]


# In[11]:


X


# In[12]:


Y


# ## Preprocess the Data

# In[13]:


le_Y = LabelEncoder()


# In[14]:


Y = le_Y.fit_transform(Y)
Y


# In[15]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[16]:


for x in range(0, X.shape[1] - 1):
    encoder_X(x)


# In[17]:


X


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [2])


# In[19]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


pd.DataFrame(pd.DataFrame(X[:, 26])[0].value_counts())


# In[21]:


ohe_X = OneHotEncoder(categorical_features = [26])


# In[22]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[59]:


sc_X = StandardScaler()


# In[60]:


X = sc_X.fit_transform(X)
X


# ## Create Train and Test data

# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[62]:


X_train.shape


# In[63]:


X_test.shape


# In[64]:


Y_train.shape


# In[65]:


Y_test.shape


# In[66]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[67]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[68]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[69]:


clf_dt.fit(X_train, Y_train)


# In[70]:


Y_pred_dt = clf_dt.predict(X_test)


# In[71]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[72]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[73]:


clf_rf.fit(X_train, Y_train)


# In[74]:


Y_pred_rf = clf_rf.predict(X_test)


# In[75]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[76]:


clf_nb = GaussianNB()


# In[77]:


clf_nb.fit(X_train, Y_train)


# In[78]:


Y_pred_nb = clf_nb.predict(X_test)


# In[79]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[80]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[81]:


clf_knn.fit(X_train, Y_train)


# In[82]:


Y_pred_knn = clf_knn.predict(X_test)


# In[83]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[84]:


clf_lr = LogisticRegression()


# In[85]:


clf_lr.fit(X_train, Y_train)


# In[86]:


Y_pred_lr = clf_lr.predict(X_test)


# In[87]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[88]:


clf_lsvc = SVC(kernel = "linear")


# In[89]:


clf_lsvc.fit(X_train, Y_train)


# In[90]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[91]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[92]:


clf_ksvc = SVC(kernel = "rbf")


# In[93]:


clf_ksvc.fit(X_train, Y_train)


# In[94]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[95]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[96]:


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




