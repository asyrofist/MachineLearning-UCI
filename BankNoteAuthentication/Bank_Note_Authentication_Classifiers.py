
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[3]:


model_accuracies = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# In[4]:


model_accuracies


# ## Importing the Data

# In[5]:


dataset = pd.read_csv("data_banknote_authentication.txt", header=None)


# In[6]:


dataset.head()


# In[7]:


dataset.shape


# In[8]:


dataset.columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]


# In[9]:


dataset.head()


# ## Creating X and Y

# In[10]:


X = dataset[["Variance", "Skewness", "Curtosis", "Entropy"]]
X.shape


# In[11]:


Y = dataset["Class"]
Y.shape


# In[12]:


X.head()


# In[13]:


Y.head()


# ## Split Data into Train and Test data

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=4)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


Y_train.shape


# In[18]:


Y_test.shape


# In[19]:


pd.DataFrame(pd.DataFrame(Y_train)['Class'].value_counts())


# In[20]:


pd.DataFrame(pd.DataFrame(Y_test)['Class'].value_counts())


# ## KNN Classifier

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[22]:


clf_knn.fit(X_train, Y_train)
Y_pred_knn = clf_knn.predict(X_test)


# In[23]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Decision Tree Classifier

# In[24]:


from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(criterion = "entropy")


# In[25]:


clf_dt.fit(X_train, Y_train)
Y_pred_dt = clf_dt.predict(X_test)


# In[26]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()


# In[28]:


clf_lr.fit(X_train, Y_train)
Y_pred_lr = clf_lr.predict(X_test)


# In[29]:


confusion_matrix(Y_test, Y_pred_lr)


# ## Random Forest Classifier

# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
clf_rf.fit(X_train, Y_train)
Y_pred_rf = clf_rf.predict(X_test)


# In[32]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes Classifier

# In[33]:


from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()


# In[34]:


clf_nb.fit(X_train, Y_train)
Y_pred_nb = clf_nb.predict(X_test)


# In[35]:


confusion_matrix(Y_test, Y_pred_nb)


# ## SVC Linear

# In[36]:


from sklearn.svm import SVC
clf_lsvc = SVC(kernel = "linear")


# In[37]:


clf_lsvc.fit(X_train, Y_train)
Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[38]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## SVC Kernel

# In[39]:


clf_ksvc = SVC(kernel = "rbf")
clf_ksvc.fit(X_train, Y_train)
Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[40]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Check the Accuracy

# In[41]:


model_accuracies


# In[42]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)


# In[43]:


model_accuracies


# In[ ]:




