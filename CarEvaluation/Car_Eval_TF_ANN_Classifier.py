
# coding: utf-8

# ## Initialize

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ## Import the Data

# In[3]:


dataset = pd.read_csv("Car.data", header = None)
dataset.shape


# In[4]:


dataset.head()


# ## Create X and Y

# In[5]:


X = dataset.iloc[:, 0:6].values
Y = dataset.iloc[:, 6].values


# In[6]:


X.shape


# In[7]:


Y.shape


# ## Preprocess the Data

# In[8]:


le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
Y = Y.reshape(len(Y), 1)

def encoder(index):
    le_X = LabelEncoder()
    X[:, index] = le_X.fit_transform(X[:, index])


# In[9]:


for i in range(0, 6):
    encoder(i)


# In[10]:


X[0]


# In[11]:


Y[0]


# In[12]:


ohe_Y = OneHotEncoder(categorical_features = [0])
Y = ohe_Y.fit_transform(Y).toarray()


# In[13]:


Y[0]


# In[14]:


Y.shape


# In[15]:


ohe_X = OneHotEncoder(categorical_features = [5])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[16]:


ohe_X = OneHotEncoder(categorical_features = [6])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[17]:


ohe_X = OneHotEncoder(categorical_features = [7])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[18]:


ohe_X = OneHotEncoder(categorical_features = [8])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[19]:


ohe_X = OneHotEncoder(categorical_features = [10])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[20]:


ohe_X = OneHotEncoder(categorical_features = [12])
X = ohe_X.fit_transform(X).toarray()
X = X[:, 1:]


# In[21]:


X.shape


# ## Create Train and Test Data

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


# ## Create TF Classifier

# In[27]:


num_features = X.shape[1]
num_features


# In[28]:


num_classes = Y.shape[1]
num_classes


# In[29]:


'''
Y = W1.X1 + ... + W4.X4 + B
output = softmax(Y)
Create Weights (W) and Biases(B) variables
'''

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))


# In[30]:


# Create x and y_ placeholders for actual data
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])


# In[31]:


# Calculate y which holds model's predicted values
Wx = tf.matmul(x, W)
y = Wx + B


# In[32]:


# Create cost function to minimize
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))


# In[33]:


# Create an optimizer to minimize the cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost_cross_entropy)


# In[38]:


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
                raise ValueError("Batch Size " + str(batch_size) + " cannot be greater than Data Size : ", len(X_train))
            else:
                batch_start_index = (i * batch_size) % (len(X_train) - batch_size)
            
            # Calculate the batch end index
            batch_end_index = batch_size + batch_start_index
            
            # Create batch x and y values
            batch_X_values = X_train[batch_start_index : batch_end_index]
            batch_Y_values = Y_train[batch_start_index : batch_end_index]
            
            # Create feed dict to feed it to optimizer
            feed = {x : np.array(batch_X_values), y_ : np.array(batch_Y_values)}
            
            sess.run(optimizer_to_use, feed_dict = feed)
            
            # Print the metrics for every 50th iteration
            if (i + 1) % 50 == 0:
                print("After " + str(i + 1) + " iterations, cost : ", sess.run(cost_cross_entropy, feed_dict = feed))
                print("W : ", sess.run(W))
                print("B : ", sess.run(B))
                print("")
                
        Y_pred_logits = sess.run(y, feed_dict = {x : np.array(X_test)})
        
        Y_pred_logits_tensor = tf.convert_to_tensor(np.array(Y_pred_logits))
        
        apply_softmax = tf.nn.softmax(Y_pred_logits_tensor)
        
        Y_pred_classes = np.argmax(sess.run(apply_softmax), axis = 1)
        
        sess.close()
    return Y_pred_classes


# In[51]:


Y_pred_classes = trainTheData(num_steps = 2000, optimizer_to_use = optimizer, batch_size = len(X_train))


# In[52]:


Y_test_classes = np.argmax(Y_test, axis = 1)


# ## Calculate the Accuracy

# In[53]:


accuracy_score(Y_pred_classes, Y_test_classes)


# In[54]:


confusion_matrix(Y_pred_classes, Y_test_classes)


# In[ ]:




