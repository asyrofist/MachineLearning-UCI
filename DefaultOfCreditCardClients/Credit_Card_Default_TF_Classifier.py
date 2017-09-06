
# coding: utf-8

# ## Initialize

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# ## Import the data

# In[3]:


dataset = pd.read_excel('Dataset.xls')
dataset.shape


# In[4]:


dataset.head()


# ## Create X and Y

# In[5]:


X = dataset.iloc[:, 0:23].values
Y = dataset.iloc[:, 23].values


# In[6]:


X.shape


# In[7]:


Y.shape


# ## Preprocess the Data

# In[8]:


le_X = LabelEncoder()
X[:, 1] = le_X.fit_transform(X[:, 1])


# In[9]:


le_X = LabelEncoder()
X[:, 2] = le_X.fit_transform(X[:, 2])


# In[10]:


le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])


# In[11]:


ohe_X = OneHotEncoder(categorical_features = [1])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[12]:


ohe_X = OneHotEncoder(categorical_features = [2])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[13]:


ohe_X = OneHotEncoder(categorical_features = [8])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[14]:


sc_X = StandardScaler()


# In[15]:


X = sc_X.fit_transform(X)


# In[16]:


X[0]


# In[17]:


X.shape


# In[18]:


ohe_Y = OneHotEncoder(categorical_features = [0])


# In[19]:


Y = Y.reshape(len(Y), 1)


# In[20]:


Y = ohe_Y.fit_transform(Y).toarray()
Y[0]


# ## Create Train and Test Data

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# In[24]:


Y_train.shape


# In[25]:


Y_test.shape


# ## Create TF Classifier

# In[26]:


num_features = X.shape[1]
num_features


# In[27]:


num_classes = Y.shape[1]
num_classes


# In[28]:


# Y = W1.X1 + W2.X2 + ... + Wn.Xn + B
# output = 1 / (1 + e ^ (- Y))
# Create Weights and Biases Variables

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))


# In[29]:


# Create x and y_ placeholders to hold actual data
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])


# In[30]:


# Calculate y which holds the model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[31]:


# Create a cost function to minimize
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))


# In[32]:


# Create an Optimizer to minimize the cost
# Use different Optimizers to compare which suits the given problem
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost_cross_entropy)


# In[33]:


def trainTheData(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # initialize the global variables
    
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
            
            # Calculate the batch X and Y Values
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dict to feed it to the optimizer
            feed = {x : np.array(batch_X_values), y_ : np.array(batch_Y_values)}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            # Print the metrics and other parameters for every 25th iteration
            if (i + 1) % 25 == 0:
                print("After " + str(i + 1) + " iterations, cost : ", sess.run(cost_cross_entropy, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
        
        # Calculate the model's predicted logit values
        Y_pred_logits = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        # Convert the logit values to a tensor
        Y_pred_logits_tensor = tf.convert_to_tensor(Y_pred_logits)
        
        # Apply softmax function
        apply_softmax = tf.nn.softmax(Y_pred_logits_tensor)
        
        # Calculate the model's predicted classes
        Y_pred_classes = np.argmax(sess.run(apply_softmax), axis = 1)
        
        sess.close()
    return Y_pred_classes   


# In[34]:


Y_pred_classes = trainTheData(num_steps = 200, optimizer_to_use = optimizer, batch_size = len(X_train))


# In[35]:


Y_test_classes = np.argmax(Y_test, axis = 1)


# ## Check the accuracy

# In[36]:


accuracy_score(Y_test_classes, Y_pred_classes)


# In[37]:


confusion_matrix(Y_test_classes, Y_pred_classes)


# In[ ]:




