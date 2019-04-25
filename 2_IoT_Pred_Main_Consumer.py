print ("Importing dependencies....")
from mapr_streams_python import Consumer, KafkaError
import os
import time
import datetime
import json
import csv
import pandas as pd
import numpy as np
import json
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import tweepy
import datetime
print("Import complete.\n")


def get_api(cfg):
  auth = tweepy.OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
  auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
  return tweepy.API(auth)

cfg = { 
    "consumer_key"        : "U7pIs5RdPsXtMvRhgslW2dq59",
    "consumer_secret"     : "oEDbf5c85RtewXagoErUvRDmKOQnn5b2JfvB4K0Fk8tz3xlZ2X",
    "access_token"        : "955555344905682944-3g079WK0HGNDpmZGAOCiySQO0VEBI79",
    "access_token_secret" : "GcNvT8dGdDgExkqkGl8QYQaOAgl6kiuIERMqhP3rqmU7c" 
    }
api = get_api(cfg)

stream_tokens  = tls.get_credentials_file()['stream_ids']
token_1 = stream_tokens[0]   # I'm getting my stream tokens from the end to ensure I'm not reusing tokens
token_2 = stream_tokens[1]

stream_id1 = dict(token=token_1, maxpoints=60)
stream_id2 = dict(token=token_2, maxpoints=60)

trace1 = go.Scatter(x=[],y=[],mode='lines',
                    line = dict(color = ('#46e2f0'),width = 4),
                    stream=stream_id1, 
                    name='Sensor')

trace2 = go.Scatter(x=[],y=[],mode='markers',
                    stream=stream_id2, 
                    marker=dict(color='#df1d1e',size=14),
                    name = 'Prediction')

data = [trace1, trace2]
layout = go.Layout(
    images= [
        dict(
         source= "https://raw.githubusercontent.com/JustinBurg/IoT_Predictive_Maintenance_Demo/master/mapr_logo.png",
         xref= "paper", yref= "paper", x= 0.389, y= 1.03, sizex= 0.1, sizey= 0.1, xanchor= "right",
         yanchor= "bottom"),
        dict(
         source= "https://raw.githubusercontent.com/JustinBurg/IoT_Predictive_Maintenance_Demo/master/mapr_logo.png",
         xref= "paper", yref= "paper", x= 0.80, y= 1.03, sizex= 0.1, sizey= 0.1, xanchor= "right", yanchor= "bottom")],
    title='Stream from Device 1',
    font=dict(family='Roboto, monospace', size=30, color='white'),
    plot_bgcolor='rgba(8,8,8,1)',
    paper_bgcolor='rgba(8,8,8,1)',
    xaxis=dict(
        domain=[0, 1.00], title='Time Period',
        titlefont=dict(family='Roboto, monospace',size=18,color='white'),
        tickfont=dict(family='Roboto, monospace',size=16,color='white')),
    yaxis=dict(
        domain=[0, 1.00], title='Measurement',
        titlefont=dict(family='Roboto, monospace',size=18,color='white'),
        tickfont=dict(family='Roboto, monospace',size=16,color='white'))) 
    

fig = go.Figure(data=data, layout=layout)
#py.iplot(fig,fileopt='overwrite', filename='IoT Robot Defect Demo')

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
DIR="/user/mapr/TFmodel/"

s_1 = py.Stream(stream_id=token_1)
s_2 = py.Stream(stream_id=token_2)

s_1.open()
s_2.open()

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
              #s_1.write(dict(x=x1,y=y1))
              #s_2.write(dict(x=x2,y=y2))
          else:
              total_list_for_RNN.append((df["Total"].iloc[-1]))
              total_metric_array = np.array(total_list_for_RNN)
              predicted_value = rnn_model(total_metric_array, num_periods)
              x1 = df["TimePeriod"].iloc[-1]
              y1 = int(df["Total"].iloc[-1])
              x2 = df["TimePeriod"].iloc[-1] + 1
              y2 = int(predicted_value)
              #s_1.write(dict(x=x1,y=y1))
              #s_2.write(dict(x=x2,y=y2))
              print("Metric at time " + str(t) + ": " + str(df["Total"].iloc[-1]) + "\nNext timestamp aggregate metric prediction: " + str(predicted_value))
              if (-200 <= predicted_value <= 450):
                  print ("Forecast does not exceed threshold for alert!\n")
              else:
                  print ("Forecast exceeds acceptable threshold - Alert Sent!\n")

              #### example of tweeting an alert with a photo ###
              if t==120:
                now = datetime.datetime.now()
                tweet = "Alert sent at " + str(now)
                image = '/user/mapr/858191982.jpg'
                status = api.update_with_media(image, status=tweet) 

              del total_list_for_RNN[0]
          time.sleep(.500)
      elif msg.error().code() != KafkaError._PARTITION_EOF:
          print(msg.error())
          running = False
  c.close()

# Close the stream when done plotting
s_1.close()
s_2.close()

