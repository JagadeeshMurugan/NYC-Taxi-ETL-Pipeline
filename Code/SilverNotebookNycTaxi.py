# Databricks notebook source
# MAGIC %md
# MAGIC ### Data Access

# COMMAND ----------

Application (client) ID:"da199206-274e-4e86-979c-c4d3c50aeb72"
Directory (tenant) ID:"807971a1-d45a-4fb2-8adb-91f99bad00fd"
secret id (value):"ehU8Q~RQBI-mY2p..QDl5rg_Mg_i0je30h6u7dr-"

# COMMAND ----------


 
spark.conf.set("fs.azure.account.auth.type.nyctaxiprojectstorage.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.nyctaxiprojectstorage.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.nyctaxiprojectstorage.dfs.core.windows.net", "da199206-274e-4e86-979c-c4d3c50aeb72")
spark.conf.set("fs.azure.account.oauth2.client.secret.nyctaxiprojectstorage.dfs.core.windows.net", "ehU8Q~RQBI-mY2p..QDl5rg_Mg_i0je30h6u7dr-")
spark.conf.set("fs.azure.account.oauth2.client.endpoint.nyctaxiprojectstorage.dfs.core.windows.net", "https://login.microsoftonline.com/807971a1-d45a-4fb2-8adb-91f99bad00fd/oauth2/token")

# COMMAND ----------

dbutils.fs.ls("abfss://bronze@nyctaxiprojectstorage.dfs.core.windows.net/trip_data/trip-data/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Reading

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from functools import reduce

# Base path
base_path = 'abfss://bronze@nyctaxiprojectstorage.dfs.core.windows.net/trip_data/trip-data/'

# List of all possible files except the problematic one
file_paths = [
    f"{base_path}green_tripdata_2023-01.parquet",
    f"{base_path}green_tripdata_2023-02.parquet",
    # Skip March
    f"{base_path}green_tripdata_2023-04.parquet",
    f"{base_path}green_tripdata_2023-05.parquet",
    f"{base_path}green_tripdata_2023-06.parquet",
    f"{base_path}green_tripdata_2023-07.parquet",
    f"{base_path}green_tripdata_2023-08.parquet",
    f"{base_path}green_tripdata_2023-09.parquet",
    f"{base_path}green_tripdata_2023-10.parquet",
    f"{base_path}green_tripdata_2023-11.parquet",
    f"{base_path}green_tripdata_2023-12.parquet"
]

# Initialize empty list to store valid dataframes
valid_dfs = []

# Process each file individually
for file_path in file_paths:
    try:
        # Read with schema inference
        df = spark.read.format("parquet").option("header", True).load(file_path)
        
        # Select columns and cast to consistent types
        processed_df = df.select(
            col("VendorID").cast("long"),
            col("lpep_pickup_datetime").cast("timestamp"),
            col("lpep_dropoff_datetime").cast("timestamp"),
            col("store_and_fwd_flag").cast("string"),
            col("RatecodeID").cast("double"),
            col("PULocationID").cast("long"),
            col("DOLocationID").cast("long"),
            col("passenger_count").cast("double"),
            col("trip_distance").cast("double"),
            col("fare_amount").cast("double"),
            col("extra").cast("double"),
            col("mta_tax").cast("double"),
            col("tip_amount").cast("double"),
            col("tolls_amount").cast("double"),
            col("ehail_fee").cast("integer"),
            col("improvement_surcharge").cast("double"),
            col("total_amount").cast("double"),
            col("payment_type").cast("double"),
            col("trip_type").cast("double"),
            col("congestion_surcharge").cast("double")
        )
        
        valid_dfs.append(processed_df)
        print(f"Successfully loaded: {file_path}")
    except Exception as e:
        print(f"Skipping {file_path}: {str(e)}")

# Combine all valid dataframes if any exist
if valid_dfs:
    df_trip_data = reduce(DataFrame.unionByName, valid_dfs)
    
    # Display schema to confirm consistent data types
    
    
    # Show sample data
    print("Sample data:")
    df_trip_data.display(5)
    
    # Cache the data for better performance in subsequent operations
    df_trip_data = df_trip_data.cache()
else:
    print("No valid files were processed successfully.")

# COMMAND ----------

# Display the total number of columns
column_count = len(df_trip_data.columns)

# Display the total number of rows
row_count = df_trip_data.count()

print(f"Total number of columns: {column_count}")
print(f"Total number of rows: {row_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data cleaning and Transformations

# COMMAND ----------

# Remove duplicate records
df_trip_data = df_trip_data.dropDuplicates()

print("Duplicates removed.")

# COMMAND ----------

# Filter invalid data points
df_trip_data = df_trip_data.filter(
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100) &
    (F.col("fare_amount") > 0) & (F.col("fare_amount") < 500)
)

print("Outliers handled.")

# COMMAND ----------

from pyspark.sql import functions as F

# Drop rows where essential fields are missing
df_trip_data = df_trip_data.dropna(subset=["lpep_pickup_datetime", "lpep_dropoff_datetime"])

# Impute numerical fields
numerical_columns = [
    "passenger_count", "trip_distance", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount", "congestion_surcharge"
]

for col in numerical_columns:
    median_value = df_trip_data.approxQuantile(col, [0.5], 0.01)[0]  # Efficient median calculation
    df_trip_data = df_trip_data.fillna({col: median_value})

# Fill nulls in categorical columns
df_trip_data = df_trip_data.fillna({"store_and_fwd_flag": "Unknown", "ehail_fee": 0})

print("Missing values handled.")

# COMMAND ----------

from pyspark.sql import functions as F

# Check for null values in each column
null_counts = df_trip_data.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_trip_data.columns]
)

# Display null counts for better insight
print("🔎 Null Count in Each Column:")
null_counts.display()

# Drop rows where **any** column is null
df_trip_data_cleaned = df_trip_data.dropna(how='any')

print(f"Null values removed successfully. Total rows remaining: {df_trip_data_cleaned.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering 

# COMMAND ----------

from pyspark.sql.functions import (col, unix_timestamp)

df_trip_data_transformed = df_trip_data_cleaned.withColumn(
    'trip_duration_min',
    (unix_timestamp(col("lpep_dropoff_datetime")) - unix_timestamp(col("lpep_pickup_datetime"))) / 60
)

# COMMAND ----------

from pyspark.sql.functions import when

df_trip_data_transformed = df_trip_data_transformed.withColumn(
    'trip_speed_mph',
    when(col('trip_duration_min') > 0, (col('trip_distance') / (col('trip_duration_min') / 60))).otherwise(0)
)

# COMMAND ----------

df_trip_data_transformed.display(5)

# COMMAND ----------

from pyspark.sql.functions import hour, dayofweek, month, year

df_trip_data_transformed = df_trip_data_transformed \
    .withColumn('hour_of_day', hour(col('lpep_pickup_datetime'))) \
    .withColumn('day_of_week', dayofweek(col('lpep_pickup_datetime'))) \
    .withColumn('month', month(col('lpep_pickup_datetime'))) \
    .withColumn('year', year(col('lpep_pickup_datetime')))

df_trip_data_transformed.display()


# COMMAND ----------

dbutils.fs.ls("abfss://bronze@nyctaxiprojectstorage.dfs.core.windows.net/")

# COMMAND ----------

# Reading 'trip_type' data
df_trip_type = spark.read.format("csv") \
    .load("abfss://bronze@nyctaxiprojectstorage.dfs.core.windows.net/trip_type/")

# Reading 'trip_zone' data
df_trip_zone = spark.read.format("csv") \
    .load("abfss://bronze@nyctaxiprojectstorage.dfs.core.windows.net/trip_zone/")

# Display sample data
df_trip_type.show(5)
df_trip_zone.show(5)

# COMMAND ----------

df_trip_type.write.format("parquet")\
    .mode("append")\
        .option("path","abfss://silver@nyctaxiprojectstorage.dfs.core.windows.net/trip_type/")\
            .save()

# COMMAND ----------

df_trip_zone.write.format("parquet")\
    .mode('append')\
        .option('path', 'abfss://silver@nyctaxiprojectstorage.dfs.core.windows.net/trip_zone/')\
            .save()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Writing to Azure Data Lake Storage

# COMMAND ----------

df_trip_data_transformed.write.format('parquet')\
    .mode('append')\
        .option('path', 'abfss://silver@nyctaxiprojectstorage.dfs.core.windows.net/trip_data/')\
            .save()

# COMMAND ----------

dbutils.fs.ls("abfss://silver@nyctaxiprojectstorage.dfs.core.windows.net/")

# COMMAND ----------

