'''
Bulk convert XML to CSV Single Partition
Author:  Justin Brandenburg, Data Scientist
Email:   jbrandenburg@mapr.com



TO START PYSPARK shell:
[justin@ip-10-0-0-122 ~]$ /opt/mapr/spark/spark-2.1.0/bin/pyspark --packages com.databricks:spark-xml_2.10:0.4.1

TO RUN PYSPARK scxript
[justin@ip-10-0-0-122 ~]$ /opt/mapr/spark/spark-2.1.0/bin/spark-submit  --packages com.databricks:spark-xml_2.10:0.4.1 /user/user01/Sensor_ETLsparksubmit.py
'''


#PYSPARK Executable script
#import libraries
print ("Importing dependencies....")
import sys
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType, DateType, TimestampType
from pyspark.sql.functions import date_format, col, desc, udf, from_unixtime, unix_timestamp, date_sub, date_add, last_day
import time
print("Import complete.\n") 

def xmlConvert(spark):

		etl_time = time.time()
        df = spark.read.format('com.databricks.spark.xml').options(rowTag='HistoricalTextData').load('iot_data')  #loads from iot_data directory in maprfs
		df = df.withColumn("TimeStamp", df["TimeStamp"].cast("timestamp")).groupBy("TimeStamp").pivot("TagName").sum("TagValue").na.fill(0)
        df.repartition(1).write.csv("iot_data_etl", header=True,sep=",")  #writes output to a single partinitioned csv
		print ("Time taken to do xml transformation: --- %s seconds ---" % (time.time() - etl_time))

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName('XML ETL') \
        .getOrCreate()

    print('Session created')

    try:
        xmlConvert(spark)

    finally:
    	spark.stop()




