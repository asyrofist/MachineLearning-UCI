
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


df = pd.read_csv('Qualitative_Bankruptcy.data.txt', header = None)
df.shape


# In[6]:


df.head()


# ## Create X and Y

# In[7]:


X = df.iloc[:, 0:6].values
Y = df.iloc[:, 6].values


# In[8]:


Y.shape


# In[9]:


X.shape


# In[10]:


X


# In[11]:


Y


# ## Preprocess the data

# In[12]:


le_Y = LabelEncoder()


# In[13]:


Y = le_Y.fit_transform(Y)


# In[14]:


Y


# In[15]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[16]:


for i in range(0, 6):
    encoder_X(i)


# In[17]:


X


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [5])


# In[19]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


ohe_X = OneHotEncoder(categorical_features = [6])


# In[21]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[22]:


ohe_X = OneHotEncoder(categorical_features = [7])


# In[23]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[24]:


ohe_X = OneHotEncoder(categorical_features = [8])


# In[25]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[26]:


ohe_X = OneHotEncoder(categorical_features = [9])


# In[27]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


ohe_X = OneHotEncoder(categorical_features = [10])


# In[29]:


X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# ## Create Train and Test data

# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# In[33]:


Y_train.shape


# In[34]:


Y_test.shape


# In[35]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[36]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[37]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[38]:


clf_dt.fit(X_train, Y_train)


# In[39]:


Y_pred_dt = clf_dt.predict(X_test)


# In[40]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[41]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[42]:


clf_rf.fit(X_train, Y_train)


# In[43]:


Y_pred_rf = clf_rf.predict(X_test)


# In[44]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[45]:


clf_nb = GaussianNB()


# In[46]:


clf_nb.fit(X_train, Y_train)


# In[47]:


Y_pred_nb = clf_nb.predict(X_test)


# In[48]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[49]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[50]:


clf_knn.fit(X_train, Y_train)


# In[51]:


Y_pred_knn = clf_knn.predict(X_test)


# In[52]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[53]:


clf_lr = LogisticRegression()


# In[54]:


clf_lr.fit(X_train, Y_train)


# In[55]:


Y_pred_lr = clf_lr.predict(X_test)


# In[56]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[57]:


clf_lsvc = SVC(kernel = "linear")


# In[58]:


clf_lsvc.fit(X_train, Y_train)


# In[59]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[60]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[61]:


clf_ksvc = SVC(kernel = "rbf")


# In[62]:


clf_ksvc.fit(X_train, Y_train)


# In[63]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[64]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[65]:


model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)
model_accuracies


# In[ ]:




