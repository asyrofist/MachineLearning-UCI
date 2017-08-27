
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
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


dataset = pd.read_csv('amazon_cells_labelled.txt', header = None, delimiter = '\t', quoting = 3)
dataset.shape


# In[5]:


dataset.head()


# ## Preprocess the Data

# In[6]:


corpus = []


# In[9]:


for i in range(0, dataset.shape[0]):
    sent = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
    sent = sent.lower()
    sent = sent.split()
    ps = PorterStemmer()
    sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    corpus.append(sent)


# In[10]:


corpus


# ## Create Bag of Words Model

# In[17]:


cv = CountVectorizer(max_features = 1300)


# In[18]:


X = cv.fit_transform(corpus).toarray()


# In[19]:


Y = dataset.iloc[:, 1].values


# In[20]:


X.shape


# In[21]:


Y.shape


# ## Create Train and Test Data

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[25]:


Y_train.shape


# In[26]:


Y_test.shape


# In[27]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[28]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[29]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[30]:


clf_dt.fit(X_train, Y_train)


# In[31]:


Y_pred_dt = clf_dt.predict(X_test)


# In[32]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[33]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[34]:


clf_rf.fit(X_train, Y_train)


# In[35]:


Y_pred_rf = clf_rf.predict(X_test)


# In[36]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[37]:


clf_nb = GaussianNB()


# In[38]:


clf_nb.fit(X_train, Y_train)


# In[39]:


Y_pred_nb = clf_nb.predict(X_test)


# In[40]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[41]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[42]:


clf_knn.fit(X_train, Y_train)


# In[43]:


Y_pred_knn = clf_knn.predict(X_test)


# In[44]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[45]:


clf_lr = LogisticRegression()


# In[46]:


clf_lr.fit(X_train, Y_train)


# In[47]:


Y_pred_lr = clf_lr.predict(X_test)


# In[48]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[49]:


clf_lsvc = SVC(kernel = 'linear')


# In[50]:


clf_lsvc.fit(X_train, Y_train)


# In[51]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[52]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[53]:


clf_ksvc = SVC(kernel = 'rbf')


# In[54]:


clf_ksvc.fit(X_train, Y_train)


# In[55]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[56]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[57]:


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




