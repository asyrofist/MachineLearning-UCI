
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


# ## Import the Data

# In[3]:


dataset = pd.read_csv('iris.data', header = None)
dataset.shape


# In[4]:


dataset.head()


# ## Create X and Y

# In[5]:


X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values


# In[6]:


X[0]


# In[7]:


Y[0]


# ## Preprocess the Data

# In[8]:


le_Y = LabelEncoder()


# In[9]:


Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)


# In[10]:


ohe_Y = OneHotEncoder(categorical_features = [0])


# In[11]:


Y = ohe_Y.fit_transform(Y).toarray()
Y[0]


# In[12]:


sc_X = StandardScaler()


# In[13]:


X = sc_X.fit_transform(X)
X[0]


# ## Create Train and Test Data

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


# ## Create TF Classifier

# In[19]:


num_features = X.shape[1]
num_features


# In[21]:


num_classes = Y.shape[1]
num_classes


# In[22]:


'''
Y = W1.X1 + ... + W4.X4 + B
output = softmax(Y)
Create Weights (W) and Biases (B) Variables
'''

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))


# In[23]:


# Create x and y_ placeholders for holding actual data
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])


# In[24]:


# Calculate y which holds the model's predicted logit values
Wx = tf.matmul(x, W)
y = Wx + B


# In[25]:


# Create cost function to minimize
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))


# In[26]:


# Create optimizer to minimize the cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost_cross_entropy)


# In[29]:


def trainTheData(num_steps, optimizer_to_use, batch_size):
    init = tf.global_variables_initializer()
    # initialize all global variables
    
    # create a session
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(num_steps):
            
            # Calculate batch start index
            if batch_size == len(X_train):
                batch_start_index = 0
            elif batch_size > len(X_train):
                raise ValueError("Batch Size : " + str(batch_size) + " cannot be greater than Data Size : ", len(X_train))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
                
            # Calculate batch end index
            batch_end_index = batch_size + batch_start_index
            
            # Create Batch X and Y Values
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dict to feed it to the optimizer
            feed = {x : np.array(batch_X_values), y_ : np.array(batch_Y_values)}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            if (i + 1) % 50 == 0:
                print("After " + str(i + 1) + " iterations, cost : ", sess.run(cost_cross_entropy, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
        
        # Store model's predicted logit values
        Y_pred_logits = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        # Convert it to tensors
        Y_pred_logits_tensor = tf.convert_to_tensor(np.array(Y_pred_logits))
        
        # Apply softmax function
        apply_softmax = tf.nn.softmax(Y_pred_logits_tensor)
        
        # Store model's predicted classes
        Y_pred_classes = np.argmax(sess.run(apply_softmax), axis = 1)
            
        sess.close()
    return Y_pred_classes        


# In[66]:


Y_pred_classes = trainTheData(num_steps = 200, optimizer_to_use = optimizer, batch_size = len(X_train))


# In[67]:


Y_test_classes = np.argmax(Y_test, axis = 1)


# ## Check the Accuracy

# In[68]:


accuracy_score(Y_test_classes, Y_pred_classes)


# In[69]:


confusion_matrix(Y_test_classes, Y_pred_classes)


# In[ ]:




