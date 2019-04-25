
# coding: utf-8

# In[1]:

import os
import pandas as pd
import csv
import numpy as np
import random
import glob
import matplotlib
import matplotlib.pyplot as plt
import random
import subprocess
get_ipython().magic('matplotlib inline')


# In[2]:

#Import the data from MapR-FS to local node directory
os.system('mkdir iot_data_etl_local')
os.system('hadoop fs -copyToLocal iot_data_etl/part-00000-478e1636-f564-4732-a3d6-8e3188d41671.csv /home/justin/iot_data_etl_local')
os.system('ls iot_data_etl_local/')
df = pd.read_csv("/home/justin/iot_data_etl_local/part-00000-478e1636-f564-4732-a3d6-8e3188d41671.csv").sort_values(['TimeStamp'], ascending=True).reset_index()


# In[3]:

df.drop(['::[scararobot]Ax_J1.PositionCommand','::[scararobot]Ax_J1.TorqueFeedback','::[scararobot]Ax_J2.PositionCommand','::[scararobot]Ax_J2.TorqueFeedback','::[scararobot]Ax_J3.TorqueFeedback','::[scararobot]Ax_J6.TorqueFeedback','::[scararobot]ScanTimeAverage','::[scararobot]Ax_J6.PositionCommand','::[scararobot]Ax_J3.PositionCommand','index'], axis=1, inplace=True)
df['TimeStamp']=pd.to_datetime(df['TimeStamp'])
print (len(df))
df.head(5)


# In[24]:

df.tail()


# In[5]:

#plot some of the columns to get an idea of the trends over time
df.plot(x="TimeStamp", y="::[scararobot]Ax_J1.ActualPosition", kind="line")
df.plot(x="TimeStamp", y=["::[scararobot]Ax_J1.ActualPosition","::[scararobot]Ax_J3.TorqueCommand"], kind="line")
df.plot(x="TimeStamp", y=["::[scararobot]CS_Cartesian.ActualPosition[0]","::[scararobot]CS_Cartesian.ActualPosition[1]"], kind="line")


# In[25]:

df['Total']= df.select_dtypes(include=['float64','float32']).apply(lambda row: np.sum(row),axis=1)
df.tail()


# In[8]:

#convert into a time series object
ts = pd.Series(df['Total'])
ts.plot(c='b', title='RW Total Sensor Aggregation')


# In[9]:

#prepare data and inputs for our TF model
num_periods = 100
f_horizon = 1       #number of periods into the future we are forecasting
TS = np.array(ts)   #convert time series object to an array
print (TS[0:10])
print (len(TS))


# In[36]:

#create our training input data set "X"
x_data = TS[:(len(TS)-(len(TS) % num_periods))]
print (x_data[0:5])
x_batches = x_data.reshape(-1, num_periods, 1)
print (len(x_batches))
print (x_batches.shape)


# In[37]:

#create our training output dataset "y"
y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
#print (y_data)
#print (len(y_data))
#y_data = TS[(num_periods+(f_horizon-1))::(num_periods)]
print (y_data)
print (len(y_data))
y_batches = y_data.reshape(-1, num_periods, 1)
print (len(y_batches))


# In[38]:

#create our test X and y data
def test_data(series,forecast,num_periods):
    test_x_setup = series[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods)
print (X_test.shape)
print (X_test[:,(num_periods-1):num_periods])
print (Y_test.shape)
print (Y_test[:,(num_periods-1):num_periods])


# In[13]:

#import tensorflow libraries
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn


# In[14]:

#set up our TF model parameters

tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

inputs = 1            #number of vectors submitted
hidden = 100          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs], name = "X")   #create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output], name = "y")


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum
#tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y),name='loss')    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()

epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
saver = tf.train.Saver()   #we are going to save the model
DIR="/home/justin/TFmodel"  #path where the model will be saved


# In[15]:

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse) 
            
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred[:,(num_periods-1):num_periods])
    saver.save(sess, os.path.join(DIR,"IoT_TF_model"),global_step = epochs)


# In[35]:

prediction_df = pd.DataFrame(list(zip(Y_test,y_pred)),columns=['ytest','ypred'])
print (prediction_df.tail(25))


# In[16]:

#Plot our test y data and our y-predicted forecast
plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
plt.show()


# In[ ]:



