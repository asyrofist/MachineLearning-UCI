
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


df = pd.read_csv('iris.data', header = None)
df.shape


# In[6]:


df.head()


# In[7]:


df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']


# In[8]:


df.head()


# ## Creating X and Y

# In[9]:


X = df.iloc[:, 0:4].values
Y = df.iloc[:, 4].values


# In[10]:


X[0]


# In[11]:


Y[0]


# ## Preprocess the Data

# In[12]:


le_Y = LabelEncoder()


# In[13]:


Y = le_Y.fit_transform(Y)
Y[0]


# In[14]:


sc_X = StandardScaler()


# In[15]:


X = sc_X.fit_transform(X)
X[0]


# ## Create Train and Test data

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[17]:


X_train.shape


# In[18]:


X_test.shape


# In[19]:


Y_train.shape


# In[20]:


Y_test.shape


# In[21]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[22]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## Decision Tree Classifier

# In[23]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[24]:


clf_dt.fit(X_train, Y_train)


# In[25]:


Y_pred_dt = clf_dt.predict(X_test)


# In[26]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest Classifier

# In[27]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[28]:


clf_rf.fit(X_train, Y_train)


# In[29]:


Y_pred_rf = clf_rf.predict(X_test)


# In[30]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes Classifier

# In[31]:


clf_nb = GaussianNB()


# In[32]:


clf_nb.fit(X_train, Y_train)


# In[33]:


Y_pred_nb = clf_nb.predict(X_test)


# In[34]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN Classifier

# In[35]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[36]:


clf_knn.fit(X_train, Y_train)


# In[37]:


Y_pred_knn = clf_knn.predict(X_test)


# In[38]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[39]:


clf_lr = LogisticRegression()


# In[40]:


clf_lr.fit(X_train, Y_train)


# In[41]:


Y_pred_lr = clf_lr.predict(X_test)


# In[42]:


confusion_matrix(Y_test, Y_pred_lr)


# ## SVC Linear

# In[43]:


clf_lsvc = SVC(kernel = "linear")


# In[44]:


clf_lsvc.fit(X_train, Y_train)


# In[45]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[46]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## SVC Kernel

# In[47]:


clf_ksvc = SVC(kernel = "rbf")


# In[48]:


clf_ksvc.fit(X_train, Y_train)


# In[49]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[50]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Classifiers

# In[51]:


model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] = accuracy_score(Y_test, Y_pred_ksvc)
model_accuracies['LinearSVC'] = accuracy_score(Y_test, Y_pred_lsvc)
model_accuracies['LogReg'] = accuracy_score(Y_test, Y_pred_lr)
model_accuracies['NB'] = accuracy_score(Y_test, Y_pred_nb)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies


# In[ ]:





# In[ ]:




