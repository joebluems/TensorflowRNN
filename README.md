# TensorflowRNN
This is a brief demo to illustrate the process of building a "simple" RNN in Tensorflow, then deploying that in real-time using MapR Stream (python client).
<br>
Note - this is based on the original blog by Justin Brandenburg: https://mapr.com/blog/streaming-predictive-maintenance-for-iot-using-tensorflow-part-1/

## Requirements
- MapR: Version 6.1
- install mapr-kafka-rest & mapr-librdkafka
- Python: 3.6.3 and ... matplotlib, pandas, tensorflow, mapr-streams-python
- Installation of mapr_streams_python may require use of global options...
- LD_LIBRARY_PATH should have librdkafka: e.g. export LD_LIBRARY_PATH=/opt/mapr/lib
 
## Setup
First, pull the code and work through the jupyter notebook that trains the model. Then, create a MapR stream & topic so you can produce data and score the model in the consumer. When you see **your_user** that means you should change the command to use your specific user name.
### Clone the repo and build the model
- cd /user/**your_user**
- git clone https://github.com/joebluems/TensorflowRNN.git
- cd TensorflowRNN
- echo $LD_LIBRARY_PATH
- jupyter notebook (play each cell)
- output should go to the folder ./rwTFmodel

### Setup a topic in MapR Streams
- maprcli stream create -path /user/**your_user**/iot_stream -produceperm p -consumeperm p -topicperm p
- maprcli stream topic create -path /user/**your_user**/iot_stream -topic sensor_record

### Start the producer and then Consumer in two terminals...
- In window #1: python 1_IoT_Pred_Main_Producer.py /user/**your_user**/iot_stream
- In window #2: python 2_IoT_Pred_Main_Consumer.py /user/**your_user**/iot_stream:sensor_record
- After the consumer hits 100 records ingested, you should start to see predicted values...
