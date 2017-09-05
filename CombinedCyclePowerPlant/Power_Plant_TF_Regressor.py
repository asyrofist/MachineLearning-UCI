
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[29]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ## Import the Data

# In[3]:


dataset = pd.read_excel('Folds5x2_pp.xlsx')
dataset.shape


# In[4]:


dataset.head()


# ## Create X and Y

# In[5]:


X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values


# In[6]:


X


# In[7]:


Y


# ## Preprocess the Data

# In[8]:


sc_X = StandardScaler()


# In[9]:


X = sc_X.fit_transform(X)


# In[10]:


X


# ## Create Train and Test Data

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[12]:


X_train.shape


# In[13]:


Y_train.shape


# In[14]:


X_test.shape


# In[15]:


Y_test.shape


# ## Create TF Regression Model

# In[16]:


num_features = X.shape[1]
num_features


# In[18]:


# Create Weights and Biases Variable
W = tf.Variable(tf.zeros([num_features, 1]))
B = tf.Variable(tf.zeros([1]))


# In[19]:


# Create x and y_ placeholders for actual values in the dataset
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, 1])


# In[20]:


# y holds model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[21]:


# Create a cost function to minimize
cost = tf.reduce_mean(tf.square(y - y_))


# In[23]:


# Create an Optimizer to minimize cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)


# In[27]:


def trainTheDataset(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # initialize all the global variables
    
    with tf.Session() as sess:
        sess.run(init)
                
        for i in range(num_steps):
            
            # Calculate batch start index
            if batch_size == len(X_train):
                batch_start_index = 0
            elif batch_size > len(X_train):
                raise ValueError("Batch Size : " + str(batch_size) + " cannot be greater than the Data Size : " + str(len(X_train)))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
                
            # Calculate batch end index
            batch_end_index = batch_start_index + batch_size
            
            # Calculate batch X and Y Values
            batch_X_Values = X_train[batch_start_index : batch_end_index]
            batch_Y_Values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dictionary to feed the optimizer
            feed = {x : np.array(batch_X_Values), y_ : np.transpose(np.array([batch_Y_Values]))}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            # Print out the cost, W, B for every 2nd Iteration
            if (i + 1) % 2 == 0:
                print("After " + str(i) + " iterations, Cost : ", sess.run(cost, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
            
        # Calculate the would be predicted values by the model
        Y_pred = sess.run(y, feed_dict = {x : np.array(X_test)})
            
        # Close the Session
        sess.close()
        
        # Return the predicted values for Test Set
        return Y_pred


# In[32]:


Y_pred = trainTheDataset(num_steps = 500, optimizer_to_use = optimizer, batch_size = len(X_train))


# ## Check the Regression metrics

# In[33]:


mean_squared_error(Y_pred, Y_test)


# In[34]:


r2_score(Y_pred, Y_test)


# In[ ]:




