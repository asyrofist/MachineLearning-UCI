
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


df = pd.read_excel('Blogger.xlsx')
df.shape


# In[6]:


df.head()


# ## Create X and Y

# In[7]:


X = df.iloc[:, 0:5].values
Y = df.iloc[:, 5].values


# ## Preprocess the data

# In[8]:


le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y


# In[9]:


le0 = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()

X[:, 0] = le0.fit_transform(X[:, 0])
X[:, 1] = le1.fit_transform(X[:, 1])
X[:, 2] = le2.fit_transform(X[:, 2])
X[:, 3] = le_Y.transform(X[:, 3])
X[:, 4] = le_Y.transform(X[:, 4])
X


# In[10]:


ohe0 = OneHotEncoder(categorical_features = [0])
X = ohe0.fit_transform(X).toarray()
X = X[:, 1:]


# In[11]:


ohe1 = OneHotEncoder(categorical_features = [2])
X = ohe1.fit_transform(X).toarray()
X = X[:, 1:]


# In[12]:


ohe1 = OneHotEncoder(categorical_features = [4])
X = ohe1.fit_transform(X).toarray()
X = X[:, 1:]
X


# ## Create Train and Test data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


Y_train.shape


# In[17]:


Y_test.shape


# In[18]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[19]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[20]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[21]:


clf_dt.fit(X_train, Y_train)


# In[22]:


Y_pred_dt = clf_dt.predict(X_test)


# In[23]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[24]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[25]:


clf_rf.fit(X_train, Y_train)


# In[26]:


Y_pred_rf = clf_rf.predict(X_test)


# In[27]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[28]:


clf_nb = GaussianNB()


# In[29]:


clf_nb.fit(X_train, Y_train)


# In[30]:


Y_pred_nb = clf_nb.predict(X_test)


# In[31]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[32]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[33]:


clf_knn.fit(X_train, Y_train)


# In[34]:


Y_pred_knn = clf_knn.predict(X_test)


# In[35]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[36]:


clf_lr = LogisticRegression()


# In[37]:


clf_lr.fit(X_train, Y_train)


# In[38]:


Y_pred_lr = clf_lr.predict(X_test)


# In[39]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[40]:


clf_lsvc = SVC(kernel = "linear")


# In[41]:


clf_lsvc.fit(X_train, Y_train)


# In[42]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[43]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[44]:


clf_ksvc = SVC(kernel = "rbf")


# In[45]:


clf_ksvc.fit(X_train, Y_train)


# In[46]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[47]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[48]:


model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)
model_accuracies


# In[ ]:




