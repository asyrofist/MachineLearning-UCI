
# coding: utf-8

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ## Importing the Data

# In[3]:


dataset = pd.read_csv('data_banknote_authentication.txt', header = None)
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


Y = Y.reshape(len(Y), 1)
Y


# In[9]:


ohe_Y = OneHotEncoder(categorical_features = [0])


# In[10]:


Y = ohe_Y.fit_transform(Y).toarray()
Y


# ## Create Train and Test Data

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 4)


# In[12]:


X_train.shape


# In[13]:


Y_train.shape


# In[14]:


X_test.shape


# In[15]:


Y_test.shape


# ## Create TF Classifier

# In[16]:


num_features = X.shape[1]
num_features


# In[17]:


num_classes = Y.shape[1]
num_classes


# In[18]:


# Y = W1.X1 + W2.X2 + ... + Wn.Xn + B
# output = 1 / (1 + e ^ (- Y))
# Create Weights and Biases Variables

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))


# In[19]:


# Create x and y_ placeholders to hold actual data
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])


# In[20]:


# Calculate y which holds the model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[21]:


# Create a cost function to minimize
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))


# In[22]:


# Create an Optimizer to minimize the cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost_cross_entropy)


# In[23]:


def trainTheData(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # Initialize all the global variables
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(num_steps):
            
            # Calculate the batch start index
            if batch_size == len(X_train):
                batch_start_index = 0
            elif batch_size > len(X_train):
                raise ValueError("Batch Size : " + str(batch_size) + " cannot be greater than Data Size : ", len(X_train))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
            
            # Calculate the batch end index
            batch_end_index = batch_size + batch_start_index
            
            # Create batch X and Y values
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dict to feed the optimizer
            feed = {x : np.array(batch_X_values), y_ : np.array(batch_Y_values)}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            if (i + 1) % 50 == 0:
                print("After " + str(i + 1) + " iterations, cross_entropy : ", sess.run(cost_cross_entropy, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
        
        # Calculate the predicted values for Test Data
        Y_pred = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        # Convert the logits into tensors
        Y_pred_tensors = tf.convert_to_tensor(np.array(Y_pred))
        
        # Apply softmax function to the tensors of logits
        apply_softmax = tf.nn.softmax(Y_pred_tensors)
        
        # Calculate the class to which it belongs
        Y_pred = np.argmax(sess.run(apply_softmax), axis = 1)
        
        sess.close()
    return Y_pred


# In[24]:


Y_pred_classes = trainTheData(num_steps = 500, optimizer_to_use = optimizer, batch_size = len(X_train))


# In[25]:


Y_pred_classes


# ## Check the accuracy

# In[26]:


Y_test_classes = np.argmax(Y_test, axis = 1)


# In[27]:


accuracy_score(Y_test_classes, Y_pred_classes)


# In[28]:


confusion_matrix(Y_test_classes, Y_pred_classes)


# In[ ]:




