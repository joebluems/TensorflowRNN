# TensorflowRNN
This is 

Note: based on the original blog by Justin Brandenburg: 

## Requirements
python3, jupyter
pip install tensorflow pandas numpy matplotlib mapr_streams_python
 
## Process
### Clone the repo and build the model
export LD_LIBRARY_PATH=/opt/mapr/lib:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.212.b04-0.el7_6.x86_64/jre/lib/amd64/server/:/opt/mapr/include
cd /mapr/<cluster_name>/user/<user>
git clone https://github.com/joebluems/TensorflowRNN.git
cd TensorflowRNN
jupyter notebook

As you browse through the notebook running the commands, you will save your TF model.

### Implement the Model with MapR Streams
maprcli stream create -path /user/<user>/iot_stream -produceperm p -consumeperm p -topicperm p
maprcli stream topic create -path /user/<user>/iot_stream -topic sensor_record


