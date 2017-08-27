
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


dataset = pd.read_csv('yelp_labelled.txt', header = None, delimiter = '\t', quoting = 3)
dataset.shape


# In[5]:


dataset.head()


# ## Preprocess the Data

# In[6]:


corpus = []


# In[7]:


for i in range(0, dataset.shape[0]):
    sent = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
    sent = sent.lower()
    sent = sent.split()
    ps = PorterStemmer()
    sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    corpus.append(sent)


# In[8]:


corpus


# ## Create Bag of Words Model

# In[15]:


cv = CountVectorizer(max_features = 1500)


# In[16]:


X = cv.fit_transform(corpus).toarray()


# In[17]:


Y = dataset.iloc[:, 1].values


# In[18]:


X.shape


# In[19]:


Y.shape


# ## Create Train and Test Data

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[21]:


X_train.shape


# In[22]:


X_test.shape


# In[23]:


Y_train.shape


# In[24]:


Y_test.shape


# In[25]:


pd.DataFrame(pd.DataFrame(Y_train)[0].value_counts())


# In[26]:


pd.DataFrame(pd.DataFrame(Y_test)[0].value_counts())


# ## DecisionTree

# In[27]:


clf_dt = DecisionTreeClassifier(criterion = 'entropy')


# In[28]:


clf_dt.fit(X_train, Y_train)


# In[29]:


Y_pred_dt = clf_dt.predict(X_test)


# In[30]:


confusion_matrix(Y_pred_dt, Y_test)


# ## Random Forest

# In[31]:


clf_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[32]:


clf_rf.fit(X_train, Y_train)


# In[33]:


Y_pred_rf = clf_rf.predict(X_test)


# In[34]:


confusion_matrix(Y_pred_rf, Y_test)


# ## Naive Bayes

# In[35]:


clf_nb = GaussianNB()


# In[36]:


clf_nb.fit(X_train, Y_train)


# In[37]:


Y_pred_nb = clf_nb.predict(X_test)


# In[38]:


confusion_matrix(Y_pred_nb, Y_test)


# ## KNN

# In[39]:


clf_knn = KNeighborsClassifier(n_neighbors = 5)


# In[40]:


clf_knn.fit(X_train, Y_train)


# In[41]:


Y_pred_knn = clf_knn.predict(X_test)


# In[42]:


confusion_matrix(Y_pred_knn, Y_test)


# ## Logistic Regression

# In[43]:


clf_lr = LogisticRegression()


# In[44]:


clf_lr.fit(X_train, Y_train)


# In[45]:


Y_pred_lr = clf_lr.predict(X_test)


# In[46]:


confusion_matrix(Y_pred_lr, Y_test)


# ## Linear SVC

# In[47]:


clf_lsvc = SVC(kernel = 'linear')


# In[48]:


clf_lsvc.fit(X_train, Y_train)


# In[49]:


Y_pred_lsvc = clf_lsvc.predict(X_test)


# In[50]:


confusion_matrix(Y_pred_lsvc, Y_test)


# ## Kernel SVC

# In[51]:


clf_ksvc = SVC(kernel = 'rbf')


# In[52]:


clf_ksvc.fit(X_train, Y_train)


# In[53]:


Y_pred_ksvc = clf_ksvc.predict(X_test)


# In[54]:


confusion_matrix(Y_pred_ksvc, Y_test)


# ## Accuracy of Various Models

# In[55]:


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




