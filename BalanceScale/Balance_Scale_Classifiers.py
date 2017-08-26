
# coding: utf-8

# ## Initialization

# In[1]:

import numpy as np
import pandas as pd


# In[4]:

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


# In[5]:

model_accuracies = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# ## Importing the Data

# In[6]:

df = pd.read_csv('balance-scale.data', header = None)
df.shape


# In[7]:

df.head()


# In[8]:

df.columns = ['Class', 'LW', 'LD', 'RW', 'RD']


# In[9]:

df.head()


# ## Preprocess the Data

# In[10]:

le = LabelEncoder()


# In[11]:

encoded_Class = le.fit(df['Class'])


# In[12]:

df['e_Class'] = df['Class'].map(lambda x : encoded_Class.transform([x]))


# In[13]:

df['e_Class'] = df['e_Class'].map(lambda x : x[0])


# In[14]:

df.head()


# ## Creating X and Y

# In[21]:

Y = df['e_Class']
Y.shape


# In[16]:

Y.head()


# In[22]:

X = df[['LW', 'LD', 'RW', 'RD']]
X.shape


# In[20]:

X.head()


# ## Create Train and Test data

# In[23]:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[24]:

X_train.shape


# In[25]:

X_test.shape


# In[26]:

Y_train.shape


# In[27]:

Y_test.shape


# In[28]:

pd.DataFrame(pd.DataFrame(Y_train)['e_Class'].value_counts())


# In[29]:

pd.DataFrame(pd.DataFrame(Y_test)['e_Class'].value_counts())


# ## Decision Tree Classifier

# In[33]:

clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[34]:

clf_dt.fit(X_train, Y_train)


# In[35]:

Y_pred_dt = clf_dt.predict(X_test)


# In[36]:

cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Random Forest Classifier

# In[37]:

clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[38]:

clf_rf.fit(X_train, Y_train)


# In[39]:

Y_pred_rf = clf_rf.predict(X_test)


# In[40]:

cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[41]:

clf_nb = GaussianNB()


# In[42]:

clf_nb.fit(X_train, Y_train)


# In[43]:

Y_pred_nb = clf_nb.predict(X_test)


# In[44]:

cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## KNN Classifier

# In[45]:

clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[46]:

clf_knn.fit(X_train, Y_train)


# In[47]:

Y_pred_knn = clf_knn.predict(X_test)


# In[48]:

cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Logistic Regression

# In[49]:

clf_lr = LogisticRegression()


# In[50]:

clf_lr.fit(X_train, Y_train)


# In[51]:

Y_pred_lr = clf_lr.predict(X_test)


# In[52]:

cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## SVC Linear

# In[53]:

clf_lsvc = SVC(kernel = "linear")


# In[54]:

clf_lsvc.fit(X_train, Y_train)


# In[55]:

Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[56]:

cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[57]:

clf_ksvc = SVC(kernel = "rbf")


# In[58]:

clf_ksvc.fit(X_train, Y_train)


# In[59]:

Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[60]:

cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracy of Various Classifiers

# In[61]:

model_accuracies


# In[62]:

model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)
model_accuracies


# In[ ]:



