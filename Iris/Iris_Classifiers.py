
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


df = pd.read_csv('iris.data', header = None)
df.shape


# In[6]:


df.head()


# In[7]:


df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']


# In[8]:


df.head()


# ## Preprocess the Data

# In[9]:


le_Class = LabelEncoder()


# In[10]:


le_Class.fit(df['Class'])


# In[11]:


df['e_Class'] = df['Class'].map(lambda x : le_Class.transform([x]))


# In[12]:


df['e_Class'] = df['e_Class'].map(lambda x : x[0])


# In[13]:


df.head()


# ## Creating X and Y

# In[14]:


Y = df['e_Class']
Y.shape


# In[15]:


Y.head()


# In[16]:


X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
X.shape


# In[17]:


X.head()


# ## Create Train and Test data

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


Y_train.shape


# In[22]:


Y_test.shape


# In[23]:


pd.DataFrame(pd.DataFrame(Y_train)['e_Class'].value_counts())


# In[24]:


pd.DataFrame(pd.DataFrame(Y_test)['e_Class'].value_counts())


# ## Decision Tree Classifier

# In[25]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[26]:


clf_dt.fit(X_train, Y_train)


# In[27]:


Y_pred_dt = clf_dt.predict(X_test)


# In[28]:


cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[29]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[30]:


clf_rf.fit(X_train, Y_train)


# In[31]:


Y_pred_rf = clf_rf.predict(X_test)


# In[32]:


cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[33]:


clf_nb = GaussianNB()


# In[34]:


clf_nb.fit(X_train, Y_train)


# In[35]:


Y_pred_nb = clf_nb.predict(X_test)


# In[36]:


cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[37]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[38]:


clf_knn.fit(X_train, Y_train)


# In[39]:


Y_pred_knn = clf_knn.predict(X_test)


# In[40]:


cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[41]:


clf_lr = LogisticRegression()


# In[42]:


clf_lr.fit(X_train, Y_train)


# In[43]:


Y_pred_lr = clf_lr.predict(X_test)


# In[44]:


cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[45]:


clf_lsvc = SVC(kernel = "linear")


# In[46]:


clf_lsvc.fit(X_train, Y_train)


# In[47]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[48]:


cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[49]:


clf_ksvc = SVC(kernel = "rbf")


# In[50]:


clf_ksvc.fit(X_train, Y_train)


# In[51]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[52]:


cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[53]:


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




