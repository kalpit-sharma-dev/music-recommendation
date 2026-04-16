import os
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['_JAVA_OPTIONS'] = '--add-opens=java.base/javax.security.auth=ALL-UNNAMED'

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkTest") \
    .master("local[*]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

# Simple test
data = [("Test", 1), ("Spark", 2)]
df = spark.createDataFrame(data, ["Word", "Count"])
df.show()

spark.stop()
print("Spark test completed successfully!")