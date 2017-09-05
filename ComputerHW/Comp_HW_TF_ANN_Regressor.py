
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf


# ## Importing the Data

# In[2]:


dataset = pd.read_csv('machine.data', header = None)
dataset.shape


# In[3]:


dataset.head()


# ## Create X and Y

# In[4]:


X = dataset.iloc[:, 2:8].values
Y = dataset.iloc[:, 8].values


# In[5]:


X.shape


# In[6]:


Y.shape


# In[7]:


X


# In[8]:


Y


# ## Preprocess the Data

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc_X = StandardScaler()


# In[11]:


X = sc_X.fit_transform(X)


# In[12]:


X


# ## Create Train and Test Data

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


Y_train.shape


# In[18]:


Y_test.shape


# ## Create and train the TensorFlow ANN Regressor

# In[19]:


# Model the Linear Regression Y = W1.X1 + W2.X2 + ... + Wn.Xn + B
# Create Weights and Biases TF Variables
num_features = X.shape[1]

W = tf.Variable(tf.zeros([num_features, 1]))
B = tf.Variable(tf.zeros([1]))


# In[20]:


# Create x and y_ placeholders for train data
x = tf.placeholder(tf.float32, [None, num_features])
Wx = tf.matmul(x, W)

# y holds model's predicted values
y = Wx + B

# y_ is a placeholder for actual y values
y_ = tf.placeholder(tf.float32, [None, 1])


# In[21]:


# Create Cost Function
cost = tf.reduce_mean(tf.square(y - y_))


# In[22]:


# Create the optimizer which will minimize the cost
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# In[23]:


def trainTheData(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # Initialize all the Global Variables
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(num_steps):
            
            # Calculate the starting index for a batch
            if batch_size == len(X_train):
                batch_start_index = 0
            elif batch_size > len(X_train):
                raise ValueError("Batch Size : " + str(batch_size) + ", must be less than Data Size : " + str(len(X_train)))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
            
            # Calculate the ending index for a batch
            batch_end_index = batch_start_index + batch_size
            
            # Get the X and Y values for the batch
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create the feed dictionary to be fed into the optimizer
            feed = {x : np.array(batch_X_values), y_ : np.transpose(np.array([batch_Y_values]))}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            # Print out the cost and other values for every 2nd iteration
            if (i + 1) % 2 == 0:
                print("After "+str(i)+" Iterations, Cost : ", sess.run(cost, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
                
        # Calculate the would be predicted values for test data by the model
        Y_pred = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        # Close the session
        sess.close()
        
        # return the predicted values for Y
        return Y_pred


# In[24]:


Y_pred = trainTheData(150, optimizer, len(X_train))


# In[25]:


Y_pred


# ## Check the Relevant Metrics for the Linear Regression Model

# In[26]:


from sklearn.metrics import mean_squared_error, r2_score


# In[27]:


mean_squared_error(Y_pred, Y_test)


# In[28]:


r2_score(Y_pred, Y_test)

