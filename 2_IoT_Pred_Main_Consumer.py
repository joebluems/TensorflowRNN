print ("Importing dependencies....")
from confluent_kafka import Consumer, KafkaError
import os
import time
import datetime
import json
import csv
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import datetime
print("Import complete.\n")

def sensor_conversion(record):
    sensor_frame = pd.DataFrame()
    sensor_frame = sensor_frame.append(record,ignore_index=True)
    sensor_frame['TimeStamp']= pd.to_datetime(sensor_frame['TimeStamp'])#.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    sensor_frame.sort_values(['TimeStamp'], ascending=True)
    sensor_frame['Total']=sensor_frame.select_dtypes(include=['float64','float32']).apply(lambda row: np.sum(row),axis=1)
    if (not os.path.isfile("IoT_Data_From_Sensor.csv")):
            sensor_frame.to_csv("IoT_Data_From_Sensor.csv")     #if csv is not there, create it
    else:
        with open('IoT_Data_From_Sensor.csv', 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(sensor_frame.tail(1))   #if csv is there, append new row to file
    return (sensor_frame)


def rnn_model(array, num_periods):
    x_data = array.reshape(-1,num_periods,1)
    #print (x_data)
    tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

    inputs = 1            #number of vectors submitted
    hidden = 100          #number of neurons we will recursively work through, can be changed to improve accuracy
    output = 1            #number of output vectors

    X = tf.placeholder(tf.float32, [None, num_periods, inputs], name = "X")   #create variable objects
    y = tf.placeholder(tf.float32, [None, num_periods, output], name = "y")

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

    learning_rate = 0.001   #small learning rate so we don't overshoot the minimum
    stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results

    loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
    training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

    init = tf.global_variables_initializer()      #initialize inputs
    saver = tf.train.Saver()                      #specify saver function                

    with tf.Session() as sess:                    #start a new tensorflow session
        saver.restore(sess, os.path.join(DIR,"IoT_TF_model-1000"))    #restore model         
        y_pred = sess.run(outputs, feed_dict={X: x_data})      #load data from streams
        FORECAST = y_pred[:,(num_periods-1):num_periods]       #only print out the last prediction, which is the forecast for next period
    return (FORECAST)


c = Consumer({'group.id': 'mygroup','default.topic.config': {'auto.offset.reset': 'earliest'}})
c.subscribe(['/user/mapr/iot_stream:sensor_record'])
DIR="./rwTFmodel/"

while True:
  running = True
  df = pd.DataFrame()
  t = 0
  total_list_for_RNN = []
  num_periods = 100  #number of periods entered into batch
  while running:
      msg = c.poll(timeout=1.0)
      if msg is None:
          continue
      if not msg.error():
          t = t + 1
          sensor_timestamp = json.loads(msg.value().decode('utf-8'))
          df = df.append(sensor_conversion(sensor_timestamp))
          df['TimePeriod'] = t
          if len(df) < num_periods:
              total_list_for_RNN.append((df["Total"].iloc[-1]))
              print("Metric at time " + str(t) + ": " + str(df["Total"].iloc[-1]))
              x1 = df["TimePeriod"].iloc[-1]
              y1 = int(df["Total"].iloc[-1])
              x2 = df["TimePeriod"].iloc[-1] + 1
              y2 = 0
          else:
              total_list_for_RNN.append((df["Total"].iloc[-1]))
              total_metric_array = np.array(total_list_for_RNN)
              predicted_value = rnn_model(total_metric_array, num_periods)
              x1 = df["TimePeriod"].iloc[-1]
              y1 = int(df["Total"].iloc[-1])
              x2 = df["TimePeriod"].iloc[-1] + 1
              y2 = int(predicted_value)
              print("Metric at time " + str(t) + ": " + str(df["Total"].iloc[-1]) + "\nNext timestamp aggregate metric prediction: " + str(predicted_value))
              if (-200 <= predicted_value <= 450):
                  print ("Forecast does not exceed threshold for alert!\n")
              else:
                  print ("Forecast exceeds acceptable threshold - Alert Sent!\n")

              del total_list_for_RNN[0]
          time.sleep(.500)
      elif msg.error().code() != KafkaError._PARTITION_EOF:
          print(msg.error())
          running = False
  c.close()

