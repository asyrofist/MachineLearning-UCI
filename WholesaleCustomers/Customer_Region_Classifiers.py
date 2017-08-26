
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

# In[4]:


dataset = pd.read_csv('Wholesale customers data.csv')
dataset.shape


# In[5]:


dataset.head()


# ## Create X and Y

# In[6]:


X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values


# In[7]:


X.shape


# In[8]:


Y.shape


# In[9]:


X


# In[10]:


Y


# ## Preprocess the Data

# In[11]:


le_Y = LabelEncoder()


# In[12]:


Y = le_Y.fit_transform(Y)


# In[13]:


Y


# In[14]:


sc_X = StandardScaler()


# In[15]:


X = sc_X.fit_transform(X)


# In[16]:


X


# ## Create Train and Test Data

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


# ## DecisionTree

# In[24]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[25]:


clf_dt.fit(X_train, Y_train)


# In[26]:


Y_pred_dt = clf_dt.predict(X_test)


# In[27]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[28]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[29]:


clf_rf.fit(X_train, Y_train)


# In[30]:


Y_pred_rf = clf_rf.predict(X_test)


# In[31]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[32]:


clf_nb = GaussianNB()


# In[33]:


clf_nb.fit(X_train, Y_train)


# In[34]:


Y_pred_nb = clf_nb.predict(X_test)


# In[35]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[36]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[37]:


clf_knn.fit(X_train, Y_train)


# In[38]:


Y_pred_knn = clf_knn.predict(X_test)


# In[39]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[40]:


clf_lr = LogisticRegression()


# In[41]:


clf_lr.fit(X_train, Y_train)


# In[42]:


Y_pred_lr = clf_lr.predict(X_test)


# In[43]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[44]:


clf_lsvc = SVC(kernel = 'linear')


# In[45]:


clf_lsvc.fit(X_train, Y_train)


# In[46]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[47]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[48]:


clf_ksvc = SVC(kernel = 'rbf')


# In[49]:


clf_ksvc.fit(X_train, Y_train)


# In[50]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[51]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[52]:


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




