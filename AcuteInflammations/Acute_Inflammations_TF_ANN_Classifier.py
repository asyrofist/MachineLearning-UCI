
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

# In[5]:


dataset = pd.read_csv("diagnosis.data", header = None, delimiter = r"\s+")
dataset.shape


# In[6]:


dataset.head()


# ## Create X and Y

# In[7]:


X = dataset.iloc[:, 0:6].values
Y1 = dataset.iloc[:, 6:7].values
Y2 = dataset.iloc[:, 7:8].values


# In[8]:


X.shape


# In[10]:


Y1.shape


# In[11]:


Y2.shape


# ## Preprocess the Data

# In[13]:


le_Y = LabelEncoder()
Y1 = le_Y.fit_transform(Y1)
Y2 = le_Y.transform(Y2)


# In[14]:


Y1[0:5]


# In[15]:


Y2[0:5]


# In[16]:


Y1 = Y1.reshape(len(Y1), 1)
Y2 = Y2.reshape(len(Y2), 1)


# In[17]:


ohe_Y = OneHotEncoder(categorical_features = [0])


# In[18]:


Y1 = ohe_Y.fit_transform(Y1).toarray()
Y1[0]


# In[19]:


Y2 = ohe_Y.transform(Y2).toarray()
Y2[0]


# In[20]:


def encoder_X(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[21]:


for i in range(1, 6):
    encoder_X(i)


# In[22]:


sc_X = StandardScaler()


# In[23]:


X = sc_X.fit_transform(X)


# In[24]:


X[0]


# ## Create Train and Test Data

# In[26]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 4)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 4)


# ## Create TF Classifier

# In[27]:


num_features = X.shape[1]
num_features


# In[29]:


num_classes = Y1.shape[1]
num_classes


# In[30]:


# Y = W1.X1 + W2.X2 + ... + Wn.Xn + B
# output = 1 / (1 + e ^ (- Y))
# Create Weights and Biases Variables

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))


# In[31]:


# Create x and y_ placeholders to hold actual data
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])


# In[32]:


# Calculate y which holds the model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[33]:


# Create a cost function to minimize
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))


# In[34]:


# Create an Optimizer to minimize the cost
# Use different Optimizers to compare which suits the given problem
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost_cross_entropy)


# In[36]:


def trainTheData(num_steps, optimizer_to_use, batch_size, X_train, Y_train, X_test):
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


# In[40]:


Y1_pred_classes = trainTheData(num_steps = 200, optimizer_to_use = optimizer, batch_size = len(X1_train), X_train = X1_train, Y_train = Y1_train, X_test = X1_test)


# In[41]:


Y1_test_classes = np.argmax(Y1_test, axis = 1)


# In[44]:


Y2_pred_classes = trainTheData(num_steps = 200, optimizer_to_use = optimizer, batch_size = len(X2_train), X_train = X2_train, Y_train = Y2_train, X_test = X2_test)


# In[45]:


Y2_test_classes = np.argmax(Y2_test, axis = 1)


# ## Check the accuracy

# In[42]:


accuracy_score(Y1_test_classes, Y1_pred_classes)


# In[43]:


confusion_matrix(Y1_test_classes, Y1_pred_classes)


# In[46]:


accuracy_score(Y2_test_classes, Y2_pred_classes)


# In[47]:


confusion_matrix(Y2_test_classes, Y2_pred_classes)


# In[ ]:




