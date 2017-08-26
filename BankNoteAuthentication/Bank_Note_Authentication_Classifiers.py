
# coding: utf-8

# In[33]:

import pandas as pd
import numpy as np


# ## Initialization

# In[117]:

model_accuracies = {'KNN':1, 'LogReg':1, 'DT':1, 'RF':1, 'NB':1, 'LinearSVC':1, 'KernelSVC':1}


# In[118]:

model_accuracies


# ## Importing the Data

# In[36]:

dataset = pd.read_csv("data_banknote_authentication.txt", header=None)


# In[37]:

dataset.head()


# In[38]:

dataset.shape


# In[39]:

dataset.columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]


# In[40]:

dataset.head()


# ## Creating X and Y

# In[55]:

X = dataset[["Variance", "Skewness", "Curtosis", "Entropy"]]
X.shape


# In[56]:

Y = dataset["Class"]
Y.shape


# In[53]:

X.head()


# In[54]:

Y.head()


# ## Split Data into Train and Test data

# In[57]:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)


# In[58]:

X_train.shape


# In[59]:

X_test.shape


# In[60]:

Y_train.shape


# In[61]:

Y_test.shape


# In[125]:

pd.DataFrame(pd.DataFrame(Y_train)['Class'].value_counts())


# In[126]:

pd.DataFrame(pd.DataFrame(Y_test)['Class'].value_counts())


# ## KNN Classifier

# In[100]:

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[101]:

clf_knn.fit(X_train, Y_train)
Y_pred_knn = clf_knn.predict(X_test)


# In[102]:

cm_knn = confusion_matrix(Y_pred_knn, Y_test)
cm_knn


# ## Decision Tree Classifier

# In[97]:

from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(criterion = "entropy")


# In[98]:

clf_dt.fit(X_train, Y_train)
Y_pred_dt = clf_dt.predict(X_test)


# In[99]:

cm_dt = confusion_matrix(Y_pred_dt, Y_test)
cm_dt


# ## Logistic Regression

# In[93]:

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()


# In[94]:

clf_lr.fit(X_train, Y_train)
Y_pred_lr = clf_lr.predict(X_test)


# In[95]:

cm_lr = confusion_matrix(Y_pred_lr, Y_test)
cm_lr


# ## Random Forest Classifier

# In[103]:

from sklearn.ensemble import RandomForestClassifier


# In[104]:

clf_rf = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
clf_rf.fit(X_train, Y_train)
Y_pred_rf = clf_rf.predict(X_test)


# In[106]:

cm_rf = confusion_matrix(Y_pred_rf, Y_test)
cm_rf


# ## Naive Bayes Classifier

# In[107]:

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()


# In[108]:

clf_nb.fit(X_train, Y_train)
Y_pred_nb = clf_nb.predict(X_test)


# In[109]:

cm_nb = confusion_matrix(Y_pred_nb, Y_test)
cm_nb


# ## SVC Linear

# In[110]:

from sklearn.svm import SVC
clf_lsvc = SVC(kernel = "linear")


# In[111]:

clf_lsvc.fit(X_train, Y_train)
Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[112]:

cm_lsvc = confusion_matrix(Y_pred_lsvc, Y_test)
cm_lsvc


# ## SVC Kernel

# In[113]:

clf_ksvc = SVC(kernel = "rbf")
clf_ksvc.fit(X_train, Y_train)
Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[115]:

cm_ksvc = confusion_matrix(Y_pred_ksvc, Y_test)
cm_ksvc


# ## Accuracies of Various Classifiers

# In[119]:

model_accuracies


# In[120]:

model_accuracies['DT'] = accuracy_score(Y_pred_dt, Y_test)
model_accuracies['KNN'] = accuracy_score(Y_pred_knn, Y_test)
model_accuracies['KernelSVC'] = accuracy_score(Y_pred_ksvc, Y_test)
model_accuracies['LinearSVC'] = accuracy_score(Y_pred_lsvc, Y_test)
model_accuracies['LogReg'] = accuracy_score(Y_pred_lr, Y_test)
model_accuracies['NB'] = accuracy_score(Y_pred_nb, Y_test)
model_accuracies['RF'] = accuracy_score(Y_pred_rf, Y_test)


# In[121]:

model_accuracies


# In[ ]:



