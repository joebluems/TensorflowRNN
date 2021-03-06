print ("Importing dependencies....")
#import mapr_kafka_rest
from confluent_kafka import Producer
import random
import time
import datetime
import os
import xml.etree.ElementTree as ET
import pandas as pd
import json
import glob
import sys


# Set up some reusable values
#sensor_topic_name  = '/iot_stream:sensor_record'
path = './test'   #local path, not hadoop fs path
stream = '/user/mapr/iot_stream'
if len(sys.argv)==2: stream=sys.argv[1]

p = Producer({'streams.producer.default.stream': stream})  #hadoop fs path

def xml2df(xml_file):
    root = ET.XML(xml_file) # element tree
    all_records = [] #This is our record list which we will convert into a dataframe
    for i, child in enumerate(root): #Begin looping through our root tree
        record = {} #Place holder for our record
        for subchild in child: #iterate through the subchildren to user-agent, Ex: ID, String, Description.
            record[subchild.tag] = subchild.text #Extract the text create a new dictionary key, value pair
            all_records.append(record) #Append this record to all_records.
    return pd.DataFrame(all_records) #return records as DataFrame

records = 0
for xml_filename in glob.glob(path+"/*.xml"):
    f = open(xml_filename, 'rb').read()
    df = xml2df(f).drop_duplicates().reset_index(drop=True).sort_values(['TimeStamp'], ascending=True)
    df['TagValue']=df.TagValue.astype(float).fillna(0)
    #df['TimeStamp']=pd.to_datetime(df['TimeStamp'])
    df = df.pivot(index='TimeStamp', columns='TagName', values='TagValue').fillna(0).rename_axis(None, axis=1).reset_index()
    df['filename'] = xml_filename
    #df = df[df["::[scararobot]speed"] != 0]
    #df['TimeStamp'] = str(datetime.datetime.df['TimeStamp'])

    for index,row in df.iterrows():
        records +=1
        sensor_df_record = {}
        sensor_df_record['filename']=row["filename"]
        sensor_df_record['TimeStamp']=row['TimeStamp']
        sensor_df_record['::[scararobot]Ax_J1.ActualPosition']=row['::[scararobot]Ax_J1.ActualPosition']
        sensor_df_record['::[scararobot]Ax_J1.PositionCommand']=row['::[scararobot]Ax_J1.PositionCommand']
        sensor_df_record['::[scararobot]Ax_J1.PositionError']=row['::[scararobot]Ax_J1.PositionError']
        sensor_df_record['::[scararobot]Ax_J1.TorqueCommand']=row['::[scararobot]Ax_J1.TorqueCommand']
        sensor_df_record['::[scararobot]Ax_J1.TorqueFeedback']=row['::[scararobot]Ax_J1.TorqueFeedback']
        sensor_df_record['::[scararobot]Ax_J2.ActualPosition']=row['::[scararobot]Ax_J2.ActualPosition']
        sensor_df_record['::[scararobot]Ax_J2.PositionCommand']=row['::[scararobot]Ax_J2.PositionCommand']
        sensor_df_record['::[scararobot]Ax_J2.PositionError']=row['::[scararobot]Ax_J2.PositionError']
        sensor_df_record['::[scararobot]Ax_J2.TorqueCommand']=row['::[scararobot]Ax_J2.TorqueCommand']
        sensor_df_record['::[scararobot]Ax_J2.TorqueFeedback']=row['::[scararobot]Ax_J2.TorqueFeedback']
        sensor_df_record['::[scararobot]Ax_J3.ActualPosition']=row['::[scararobot]Ax_J3.ActualPosition']
        sensor_df_record['::[scararobot]Ax_J3.PositionCommand']=row['::[scararobot]Ax_J3.PositionCommand']
        sensor_df_record['::[scararobot]Ax_J3.PositionError']=row['::[scararobot]Ax_J3.PositionError']
        sensor_df_record['::[scararobot]Ax_J3.TorqueCommand']=row['::[scararobot]Ax_J3.TorqueCommand']
        sensor_df_record['::[scararobot]Ax_J3.TorqueFeedback']=row['::[scararobot]Ax_J3.TorqueFeedback']
        sensor_df_record['::[scararobot]Ax_J6.ActualPosition']=row['::[scararobot]Ax_J6.ActualPosition']
        sensor_df_record['::[scararobot]Ax_J6.PositionCommand']=row['::[scararobot]Ax_J6.PositionCommand']
        sensor_df_record['::[scararobot]Ax_J6.PositionError']=row['::[scararobot]Ax_J6.PositionError']
        sensor_df_record['::[scararobot]Ax_J6.TorqueCommand']=row['::[scararobot]Ax_J6.TorqueCommand']
        sensor_df_record['::[scararobot]Ax_J6.TorqueFeedback']=row['::[scararobot]Ax_J6.TorqueFeedback']
        sensor_df_record['::[scararobot]CS_Cartesian.ActualPosition[0]']=row['::[scararobot]CS_Cartesian.ActualPosition[0]']
        sensor_df_record['::[scararobot]CS_Cartesian.ActualPosition[1]']=row['::[scararobot]CS_Cartesian.ActualPosition[1]']
        sensor_df_record['::[scararobot]CS_Cartesian.ActualPosition[2]']=row['::[scararobot]CS_Cartesian.ActualPosition[2]']
        sensor_df_record['::[scararobot]CS_SCARA.ActualPosition[0]']=row['::[scararobot]CS_SCARA.ActualPosition[0]']
        sensor_df_record['::[scararobot]CS_SCARA.ActualPosition[1]']=row['::[scararobot]CS_SCARA.ActualPosition[1]']
        sensor_df_record['::[scararobot]CS_SCARA.ActualPosition[2]']=row['::[scararobot]CS_SCARA.ActualPosition[2]']
        sensor_df_record['::[scararobot]ScanTimeAverage']=row['::[scararobot]ScanTimeAverage']
        sensor_df_record['::[scararobot]speed']=row['::[scararobot]speed']


        record = json.dumps(sensor_df_record)
        p.produce('sensor_record',record)
        print("POSTED: " + str(sensor_df_record))
        time.sleep(0.25)

print("\nExecuted parsing and streaming "+ str(records) + " Timestamps!")
p.flush()
