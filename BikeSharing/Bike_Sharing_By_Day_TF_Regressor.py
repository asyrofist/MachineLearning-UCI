
# coding: utf-8

# ## Initialization

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ## Import the Data

# In[3]:


dataset = pd.read_csv('day.csv')
dataset.shape


# In[4]:


dataset.head()


# ## Create X and Y

# In[5]:


X = dataset.iloc[:, 2:14].values
Y = dataset.iloc[:, 15].values


# In[6]:


X


# In[8]:


Y


# ## Preprocess the Data

# In[9]:


sc_X = StandardScaler()


# In[10]:


X = sc_X.fit_transform(X)


# In[11]:


X


# ## Create Train and Test Data

# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


Y_train.shape


# In[16]:


Y_test.shape


# ## Create TF Regressor

# In[17]:


num_features = X.shape[1]
num_features


# In[18]:


# Create Weights and Biases Variable
W = tf.Variable(tf.zeros([num_features, 1]))
B = tf.Variable(tf.zeros([1]))


# In[19]:


# Create x and y_ placeholders for holding actual dataset values
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, 1])


# In[20]:


# y holds the model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[21]:


# Create Cost function which has to be minimized
cost = tf.reduce_mean(tf.square(y - y_))


# In[22]:


# Create an optimizer to minimize cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)


# In[23]:


def trainTheData(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # initialize the global variables
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(num_steps):
            
            # Calculate batch start index
            if batch_size == len(X_train):
                batch_start_index = 0
            elif batch_size > len(X_train):
                raise ValueError("Batch Size : " + str(batch_size) + " cannot be greater than Data Size : ",len(X_train))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
                
            # Calculate batch end index
            batch_end_index = batch_start_index + batch_size
            
            # Create Batch X and Y values
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dictionary to feed it to the optimizer
            feed = {x : np.array(batch_X_values), y_ : np.transpose(np.array([batch_Y_values]))}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            # Print out every 2nd iteration value
            if (i + 1) % 2 == 0:
                print("After " + str(i) + " iterations, Cost : ", sess.run(cost, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
                
        Y_pred = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        sess.close()
        return Y_pred        


# In[24]:


Y_pred = trainTheData(num_steps = 200, optimizer_to_use = optimizer, batch_size = len(X_train))


# ## Check the Regression Metrics

# In[25]:


mean_squared_error(Y_pred, Y_test)


# In[26]:


r2_score(Y_pred, Y_test)


# In[ ]:




