#!/usr/bin/env python
# coding: utf-8

# In[1]:


## project in Databricks
from pyspark.sql.types import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
#from pyspark.sql import functions as F
from pyspark.ml.linalg import SparseVector, DenseVector
from sklearn import linear_model
import datetime as dt
import matplotlib.pyplot as plt
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
from pyspark.sql import functions as F 
sqlContext = SQLContext
from pyspark.sql.types import *
import pandas as pd


# In[2]:


#start = dt.datetime.now()
data = spark.read.csv('FileStore/tables/nyc_taxi.csv', header=True, inferSchema=True)
data.printSchema()


# In[3]:


data.describe().toPandas().transpose()
data=data.withColumn("pickup_time",data['pickup_time'].cast("timestamp"))
data=data.withColumn("dropoff_time",data['dropoff_time'].cast("timestamp"))


# In[4]:


data=data.filter((col("distance")!=0) & (col("fare")>0))
# subtracting tips from the fare.
data=data.withColumn("Feature_Fare",(col("fare")-col("tip")))
#1 Using Spark MLlib build a model to predict taxi fare from trip distance (M1)
vectorAssembler=VectorAssembler(inputCols=['distance'],outputCol='features')
dist_df=vectorAssembler.transform(data)
dist_df=dist_df.select(['features','Feature_fare'])
dist_df.show()


# In[5]:


#Linear Regression Model building
lr = LinearRegression(featuresCol='features', labelCol='Feature_fare',maxIter=100,regParam=0.3, elasticNetParam=0.8)
lr_model= lr.fit(dist_df)
print("Intercept:"+str(lr_model.intercept))
print("Coefficients:"+str(lr_model.coefficients))
dist_df.describe().show()
lr_model.save("linreg.model1")


# In[6]:


#Calculating RMSE AND R^2
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[7]:


##2.1 What is the fare of a 20 mile long trip using M1
testdf = sc.parallelize([(1, DenseVector(20))]).toDF(["row_num", "features"])
lr_predictions = lr_model.transform(testdf)
lr_predictions.select("prediction","row_num","features").show()


# In[8]:


#2 Using Spark MLlib build a model to predict taxi fare from trip distance and trip duration in minutes (M2)
timeformat= "'T'HH:mm"
timediff=(F.unix_timestamp('dropoff_time',format=timeformat) - F.unix_timestamp('pickup_time',format=timeformat))
data=data.withColumn("Duration",timediff)
data=data.withColumn("Duration", data["Duration"]/60)
data=data.filter(col("duration")!=0)


# In[9]:


vectorAssembler2=VectorAssembler(inputCols=['distance','Duration'],outputCol='features')
duration_df=vectorAssembler2.transform(data)
duration_df=duration_df.select(['features','Feature_fare'])
duration_df.show()


# In[10]:


lr2 = LinearRegression(featuresCol='features', labelCol='Feature_fare',maxIter=100,regParam=0.3, elasticNetParam=0.8)
lr_model2= lr2.fit(duration_df)
print("Intercept:"+str(lr_model2.intercept))
print("Coefficients:"+str(lr_model2.coefficients))


# In[11]:


#Calculating RSME and r^2
trainingSummary = lr_model2.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[12]:


#2.2 What is the fare of a 14 mile trip that took 75 minutes using M2
testdf2 = sc.parallelize([(1, DenseVector([14,75]))]).toDF(["row_num", "features"])
lr_predictions2 = lr_model2.transform(testdf2)
lr_predictions2.select("prediction","features").show()


# In[13]:


#2.3Which fare is higher 10 mile trip taking 40 min or 13 mile trip taking 25 min? Use M2 to answer this question
fare1 = sc.parallelize([(1, DenseVector([10,40]))]).toDF(["row_num", "features"])
fare1_predictions = lr_model2.transform(fare1)
fare1_predictions.select("prediction","features").show()

fare2 = sc.parallelize([(1, DenseVector([13,25]))]).toDF(["row_num", "features"])
fare2_predictions = lr_model2.transform(fare2)
fare2_predictions.select("prediction","features").show()

#13 mile in 25 min has higher fare.


# In[14]:


#3.Using Spark operations (transformation and actions) compute the average tip amount
data_rdd=data.rdd
fare_rdd=data_rdd
tips= fare_rdd.map(lambda x:x[5])
mean = tips.mean()
print(mean)


# In[15]:


# https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html
#4. Most number of trips
data_rdd=data.rdd
pickup_rdd=data_rdd.map(lambda x:x[1])
pickup_split=pickup_rdd.map(lambda x:x[0:2])
splitter=pickup_split.map(lambda w:(w,1))
reduced_rdd=splitter.reduceByKey(lambda x,y:x+y)
reduced_rdd.collect()
#Maximum number of trips happen  from 17:00 - 18:00 i.e 5pm to 6 pm


# In[16]:



#df = spark.read.csv('FileStore/tables/nyc_taxi.csv', header=True, inferSchema=True)
import datetime as dt
fare_df=data.withColumn("Feature_Fare",(col("fare")))
splits = fare_df.randomSplit([0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
st=dt.datetime.now()
split1_df=vectorAssembler.transform(splits[0])
split1_df=split1_df.select(['features','Feature_fare'])
split1= lr2.fit(split1_df)


# In[17]:


ml={}
splits = fare_df.randomSplit([0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
for i in [0,1,2,3,4,5,6,7,8,9]:
  st=dt.datetime.now()
  split_df=vectorAssembler.transform(splits[i])
  split_df=split_df.select(['features','Feature_fare'])
  split_df=lr.fit(split_df)
  et=dt.datetime.now()
  ml[i]=(et-st).total_seconds()
  print("Performance for", i, "% :", ml[i])


# In[18]:


#timeformat= "'T'HH:mm"
#timediff=(F.unix_timestamp('dropoff_time',format=timeformat) - F.unix_timestamp('pickup_time',format=timeformat))
#data=data.withColumn("Duration",timediff)
#data=data.withColumn("Duration", data["Duration"]/60)
#data=data.filter(col("duration")!=0)
ml2={}
splits_1 = fare_df.randomSplit([0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
for i in [0,1,2,3,4,5,6,7,8,9]:
  st=dt.datetime.now()
  vectorAssembler3=VectorAssembler(inputCols=['distance','Duration'],outputCol='features')
  split_df_1=vectorAssembler3.transform(splits_1[i])
  split_df_1=split_df_1.select(['features','Feature_fare'])
  split_df_1=lr2.fit(split_df_1)
  et=dt.datetime.now()
  ml2[i]=(et-st).total_seconds()
  print("Performance for", i, "% :", ml_2[i])


# In[19]:


scikit = {}
from sklearn import linear_model
import datetime as dt
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    start = dt.datetime.now()        
    data =  [k.strip().split(',') for k in open('/dbfs/FileStore/tables/data/nyc_taxi.csv','r').readlines()[1:]]
    feature = []
    label = []
    for d in data[:int(len(data)*i)]:
        feature.append([float(d[-3])])
        label.append(float(d[-1]))
    reg = linear_model.LinearRegression()
    reg.fit(feature, label)
    end = dt.datetime.now()
    scikit[i] = (end-start).total_seconds()
    print("Performance for", i*100, "% :", scikit[i])


# In[20]:


scikito = {}
from sklearn import linear_model
import datetime as dt
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    start = dt.datetime.now()
    data2 =  [k.strip().split(',') for k in open('/dbfs/FileStore/tables/data/nyc_taxi.csv','r').readlines()[1:]]
    feature2 = []
    label2 = []
    for d in data2[:int(len(data2)*i)]:
        feature2.append([float(d[-3]),float(d[-4])])
        label2.append(float(d[-1]))
    reg2 = linear_model.LinearRegression()
    reg2.fit(feature, label)
    end = dt.datetime.now()
    scikito[i] = (end-start).total_seconds()
    print("Performance for", i*100, "% :", scikito[i])


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
v1 = [ v for v in ml.values() ]
v2 = [ v for v in ml_2.values() ]
v3 = [ v for v in scikit.values() ]
v4 = [ v for v in scikito.values() ]
df=pd.DataFrame({'x': range(1,11,1), 'y1': v1, 'y2': v2, 'y3':v3,'y4':v4 })
 
 # plot
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='red',color='red', linewidth=2,label="M1_Spark")
plt.plot( 'x', 'y2', data=df, marker='o', color='green', linewidth=2,label="M2_Spark")
plt.plot( 'x', 'y3', data=df, marker='o',markerfacecolor='blue', color='blue', linewidth=2, linestyle='dashed', label="M1_Scikit")
plt.plot( 'x', 'y4', data=df, marker='o', color= 'green', linewidth=2, linestyle='dashed', label="M2_Scikit")
plt.legend(loc='best')
plt.savefig('Performance.png')


# In[22]:




