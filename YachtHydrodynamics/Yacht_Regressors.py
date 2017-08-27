
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[3]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[4]:


mses = {'LinReg' : 0, 'SVR': 0, 'DTR':0, 'RFR':0, 'KNNR':0}
r2s = {'LinReg' : 0, 'SVR': 0, 'DTR':0, 'RFR':0, 'KNNR':0}


# ## Importing the Data

# In[5]:


dataset = pd.read_csv('yacht_hydrodynamics.data', delimiter = r'\s+', header = None)
dataset.shape


# In[6]:


dataset.head()


# ## Create X and Y

# In[7]:


X = dataset.iloc[:, 0:6].values
Y = dataset.iloc[:, 6].values


# In[8]:


X.shape


# In[9]:


Y.shape


# In[10]:


X


# In[11]:


Y


# ## Preprocess the Data

# In[13]:


sc_X = StandardScaler()


# In[14]:


X = sc_X.fit_transform(X)


# In[15]:


X


# ## Create Train and Test Data

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


# ## Linear Regression

# In[21]:


reg_lin = LinearRegression()


# In[22]:


reg_lin.fit(X_train, Y_train)


# In[23]:


Y_pred_lin = reg_lin.predict(X_test)


# In[25]:


plt.figure(figsize = (12,10))
plt.plot(Y_pred_lin, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.title('Linear Regressor')
plt.show()


# ## SVR

# In[26]:


reg_lsvr = SVR(kernel = 'linear')


# In[27]:


reg_lsvr.fit(X_train, Y_train)


# In[28]:


Y_pred_lsvr = reg_lsvr.predict(X_test)


# In[29]:


plt.figure(figsize = (12,10))
plt.plot(Y_pred_lsvr, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.title('SVR')
plt.show()


# ## Decision Tree Regressor

# In[30]:


reg_dtr = DecisionTreeRegressor()


# In[31]:


reg_dtr.fit(X_train, Y_train)


# In[32]:


Y_pred_dtr = reg_dtr.predict(X_test)


# In[33]:


plt.figure(figsize = (12,10))
plt.plot(Y_pred_dtr, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.title('DT Regressor')
plt.show()


# ## Random Forest Regressor

# In[34]:


reg_rfr = RandomForestRegressor(n_estimators=200)


# In[35]:


reg_rfr.fit(X_train, Y_train)


# In[36]:


Y_pred_rfr = reg_rfr.predict(X_test)


# In[37]:


plt.figure(figsize = (12,10))
plt.plot(Y_pred_rfr, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.title('RF Regressor')
plt.show()


# ## KNN Regressor

# In[38]:


reg_knnr = KNeighborsRegressor(n_neighbors = 2)


# In[39]:


reg_knnr.fit(X_train, Y_train)


# In[40]:


Y_pred_knnr = reg_knnr.predict(X_test)


# In[41]:


plt.figure(figsize = (12,10))
plt.plot(Y_pred_knnr, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.title('KNN Regressor')
plt.show()


# ## Metrics

# In[42]:


mses['LinReg'] = mean_squared_error(Y_pred_lin, Y_test)
mses['SVR'] = mean_squared_error(Y_pred_lsvr, Y_test)
mses['DTR'] = mean_squared_error(Y_pred_dtr, Y_test)
mses['RFR'] = mean_squared_error(Y_pred_rfr, Y_test)
mses['KNNR'] = mean_squared_error(Y_pred_knnr, Y_test)
mses


# In[43]:


r2s['LinReg'] = r2_score(Y_pred_lin, Y_test)
r2s['SVR'] = r2_score(Y_pred_lsvr, Y_test)
r2s['DTR'] = r2_score(Y_pred_dtr, Y_test)
r2s['RFR'] = r2_score(Y_pred_rfr, Y_test)
r2s['KNNR'] = r2_score(Y_pred_knnr, Y_test)
r2s


# In[ ]:




