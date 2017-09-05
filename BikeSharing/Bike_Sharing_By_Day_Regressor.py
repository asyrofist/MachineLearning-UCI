
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ## Importing the Data

# In[5]:


df = pd.read_csv('day.csv')
df.shape


# In[6]:


df.head()


# ## Creating X and Y

# In[10]:


X = df.iloc[:, 2:14].values
Y = df.iloc[:, 15].values


# In[11]:


X


# In[12]:


Y


# ## Preprocess the Data

# In[19]:


sc_X = StandardScaler()


# In[20]:


X = sc_X.fit_transform(X)


# In[21]:


X


# ## Creating Train and Test Data

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[23]:


X_train.shape


# In[24]:


Y_train.shape


# In[25]:


X_test.shape


# In[26]:


Y_test.shape


# ## Training the Model

# In[28]:


reg_lin = LinearRegression()


# In[29]:


reg_lin.fit(X_train, Y_train)


# In[30]:


Y_pred = reg_lin.predict(X_test)


# ## Checking the Accuracy

# In[31]:


print(mean_squared_error(Y_pred, Y_test))


# In[32]:


print(r2_score(Y_pred, Y_test))


# ## Plotting Predicted and Actual Values

# In[34]:


plt.figure(figsize = (16,10))
plt.plot(Y_pred, ms = 50, alpha = 1, color = 'red')
plt.plot(Y_test, ms = 50, alpha = 1, color = 'black')
plt.legend(['Predicted', 'Actual'], fontsize = '15')
plt.show()


# In[ ]:




