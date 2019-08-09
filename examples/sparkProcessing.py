import json
import argparse

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import col, lit

def main(path,test_path,write_location,test_write_location):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    if is_comp:
        df = spark.read.csv(path,header=True).cache()
        npart = 10000
    else:
        # define schema
        structs = [StructField("Label", IntegerType(), True)]

        for dfeat in dense_features:
            structs.append(StructField(dfeat,DoubleType(), True))

        for cfeat in sparse_features:
            structs.append(StructField(cfeat,StringType(), True))

        schema = StructType(structs)
        
        df1 = spark.read.csv(path+"/day_[0-9]",sep="\t",schema=schema)
        df2 = spark.read.csv(path+"/day_1[0-9]",sep="\t",schema=schema)
        
        tcolumns = df1.columns
        df = df1.select(tcolumns).union(df2.select(tcolumns))
        npart = 860000


    print("Num examples: ",df.count())

    # transformation training data

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
        print("dense: ",col_t)
        min_t = df.agg({col_t:"min"}).collect()[0][0]
        max_t = df.agg({col_t:"max"}).collect()[0][0]
        dense_meta[col_t] = [min_t, max_t]
        df = df.withColumn(col_t+"_scaled",(col(col_t)-min_t)/(max_t-min_t)*(scaled_max-scaled_min)+scaled_min)
        df = df.drop(col_t).withColumnRenamed(col_t+"_scaled",col_t)

    ## index categoricals
    indexers = {}
    for col_t in sparse_features:
        print("cat:",col_t)
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
        
    cat_meta = {}
    for xk in indexers.keys():
        x = indexers[xk]
        cat_meta[xk] = len(x.labels)
        
    json.dump(cat_meta,open("categorical-meta.json","w"))
    json.dump(all_index,open("categorical.json",'w'))
    json.dump(dense_meta,open("dense-meta.json",'w'))


    # (optional) store in s3
    # Insert uploading code here

    # save training data
    df = df.repartition(npart)
    df.write.mode("overwrite").csv(write_location,header=True)

    # read test data
    if is_comp:
        df = spark.read.csv(test_path,header=True)
    else:
        df = spark.read.csv(test_path,sep="\t",schema=schema)
    # transform test data
    # change datatype of dense features
    for col_t in dense_features:
        df = df.withColumn(col_t,col(col_t).cast(DoubleType()))

    ## fill nulls
    df = df.fillna('NULL',subset=sparse_features)
    df = df.fillna(0.,subset=dense_features)

    # use already computed statistics
    ## dense features
    dense_meta = {}
    for col_t in dense_features:
        min_t = dense_meta[col_t][0]
        max_t = dense_meta[col_t][1]
        dense_meta[col_t] = [min_t, max_t]
        df = df.withColumn(col_t+"_scaled",(col(col_t)-min_t)/(max_t-min_t)*(scaled_max-scaled_min)+scaled_min)
        df = df.drop(col_t).withColumnRenamed(col_t+"_scaled",col_t)

    ## index categoricals
    indexers = {}
    for col_t in sparse_features:
        fitted_indexer = indexers[col_t]
        df = fitted_indexer.transform(df)
        indexers[col_t] = fitted_indexer # save indexer for test data
        df = df.drop(col_t).withColumnRenamed(col_t+"_indexed",col_t)
        df = df.withColumn(col_t,col(col_t).cast(IntegerType()))

    # convert label dtype
    df = df.withColumn("Label",col("Label").cast(DoubleType()))

    df = df.repartition(npart)
    df.write.mode("overwrite").csv(test_write_location,header=True)

if __name__ == '__main__':
    main(test_write_location)