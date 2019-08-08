"""Script for processing criteo datasets.
"""
import json

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler

from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, lit

# keep the file sizes smaller for large distribution experiments
spark.conf.set("spark.sql.files.maxRecordsPerFile", 100000)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

df = spark.read.csv(path,header=True).cache()

print("Number of examples: ",df.count())

# change datatype of dense features
for col_t in dense_features:
  df = df.withColumn(col_t,col(col_t).cast(DoubleType()))

## fill nulls
df = df.fillna('NULL',subset=sparse_features)
df = df.fillna(0.,subset=dense_features)

# compute statistics
## dense features
scaled_max = 1
scaled_min = 0
dense_meta = {}
for col_t in dense_features:
  min_t = df.agg({col_t:"min"}).collect()[0][0]
  max_t = df.agg({col_t:"max"}).collect()[0][0]
  dense_meta[col_t] = [min_t, max_t]
  df = df.withColumn(col_t+"_scaled",(col(col_t)-min_t)/(max_t-min_t)*(scaled_max-scaled_min)+scaled_min)
  df = df.drop(col_t).withColumnRenamed(col_t+"_scaled",col_t)

## index categoricals
indexers = {}
for col_t in sparse_features:
  indexer = StringIndexer(inputCol=col_t, outputCol=col_t+"_indexed") 
  fitted_indexer = indexer.fit(df)
  df = fitted_indexer.transform(df)
  indexers[col_t] = fitted_indexer # save indexer for test data
  df = df.drop(col_t).withColumnRenamed(col_t+"_indexed",col_t)
  df = df.withColumn(col_t,col(col_t).cast(IntegerType()))

# convert label dtype
df = df.withColumn("Label",col("Label").cast(DoubleType()))


# save statistics/meta data locally
all_index = {}
for xk in indexers.keys():
  x = indexers[xk]

  index2name = dict([y for y in zip(range(len(x.labels)),x.labels)])
  name2index = {v: k for k, v in index2name.items()}
  
  all_index[xk] = {'index2name':index2name,
                   'name2index':name2index}
  
json.dump(all_index,open("categorical.json",'w'))
json.dump(dense_meta,open("dense-meta.json",'w'))


# save processed training data
df = df.repartition(1000)
df.write.mode("overwrite").csv(write_location,header=True)