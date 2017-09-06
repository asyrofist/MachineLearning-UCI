
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


dataset = pd.read_csv('bank.csv', delimiter=';', quoting=3)
dataset.shape


# In[5]:


dataset = dataset.rename(columns = lambda x : x.replace('"', ''))
dataset.head()


# In[6]:


columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for x in columns:
    dataset[x] = dataset[x].apply(lambda x : x.replace('"', ''))


# In[7]:


dataset.head()


# In[8]:


dataset['age'] = pd.to_numeric(dataset['age'])


# ## Create X and Y

# In[9]:


X = dataset.iloc[:, 0:16].values
Y = dataset.iloc[:, 16].values


# In[10]:


X.shape


# In[11]:


Y.shape


# In[12]:


X


# In[13]:


Y


# ## Preprocess the Data

# In[14]:


le_Y = LabelEncoder()


# In[15]:


Y = le_Y.fit_transform(Y)
Y


# In[16]:


cols_to_encode = [1, 2, 3, 4, 6, 7, 8, 10, 15]
def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])
    return


# In[17]:


for i in cols_to_encode:
    encoder_X(i)


# In[18]:


X[0, :]


# In[19]:


ohe_X = OneHotEncoder(categorical_features = [15])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[20]:


pd.DataFrame(pd.DataFrame(X[:, 13])[0].value_counts())


# In[21]:


ohe_X = OneHotEncoder(categorical_features = [13])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[22]:


pd.DataFrame(pd.DataFrame(X[:, 22])[0].value_counts())


# In[23]:


ohe_X = OneHotEncoder(categorical_features = [22])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[24]:


pd.DataFrame(pd.DataFrame(X[:, 19])[0].value_counts())


# In[25]:


ohe_X = OneHotEncoder(categorical_features = [19])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[26]:


pd.DataFrame(pd.DataFrame(X[:, 21])[0].value_counts())


# In[27]:


ohe_X = OneHotEncoder(categorical_features = [21])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[28]:


pd.DataFrame(pd.DataFrame(X[:, 22])[0].value_counts())


# In[29]:


ohe_X = OneHotEncoder(categorical_features = [22])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]
X.shape


# In[30]:


sc_X = StandardScaler()


# In[31]:


X = sc_X.fit_transform(X)


# In[32]:


X


# ## Create Train and Test Data

# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[34]:


X_train.shape


# In[35]:


X_test.shape


# In[36]:


Y_train.shape


# In[37]:


Y_test.shape


# In[38]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[39]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[40]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[41]:


clf_dt.fit(X_train, Y_train)


# In[42]:


Y_pred_dt = clf_dt.predict(X_test)


# In[43]:


confusion_matrix(Y_test, Y_pred_dt)


# ## Random Forest

# In[44]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[45]:


clf_rf.fit(X_train, Y_train)


# In[46]:


Y_pred_rf = clf_rf.predict(X_test)


# In[47]:


confusion_matrix(Y_test, Y_pred_rf)


# ## Naive Bayes

# In[48]:


clf_nb = GaussianNB()


# In[49]:


clf_nb.fit(X_train, Y_train)


# In[50]:


Y_pred_nb = clf_nb.predict(X_test)


# In[51]:


confusion_matrix(Y_test, Y_pred_nb)


# ## KNN

# In[52]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[53]:


clf_knn.fit(X_train, Y_train)


# In[54]:


Y_pred_knn = clf_knn.predict(X_test)


# In[55]:


confusion_matrix(Y_test, Y_pred_knn)


# ## Logistic Regression

# In[56]:


clf_lr = LogisticRegression()


# In[57]:


clf_lr.fit(X_train, Y_train)


# In[58]:


Y_pred_lr = clf_lr.predict(X_test)


# In[59]:


confusion_matrix(Y_test, Y_pred_lr)


# ## Linear SVC

# In[60]:


clf_lsvc = SVC(kernel = 'linear')


# In[61]:


clf_lsvc.fit(X_train, Y_train)


# In[62]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[63]:


confusion_matrix(Y_test, Y_pred_lsvc)


# ## Kernel SVC

# In[64]:


clf_ksvc = SVC(kernel = 'rbf')


# In[65]:


clf_ksvc.fit(X_train, Y_train)


# In[66]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[67]:


confusion_matrix(Y_test, Y_pred_ksvc)


# ## Accuracy of Various Models

# In[68]:


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




