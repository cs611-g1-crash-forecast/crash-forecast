# Checks that the image works: Spark starts on Java, the ML libraries import,
# the raw data is visible and read-only, and datamart/ is writable.
import os
import platform

import pandas
import pyspark
import shapely
import sklearn
import xgboost
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("env_check").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("python", platform.python_version())
    print("java", spark.sparkContext._jvm.java.lang.System.getProperty("java.version"))
    print("pyspark", pyspark.__version__, "| cores:", spark.sparkContext.defaultParallelism)
    print("pandas", pandas.__version__, "| scikit-learn", sklearn.__version__,
          "| xgboost", xgboost.__version__, "| shapely", shapely.__version__)

    # a small Spark job: sum 0..999,999
    total = spark.range(1_000_000).selectExpr("sum(id) AS total").collect()[0]["total"]
    print("spark job: sum of 0..999,999 =", total)

    # the raw data, mounted read-only from ../data
    for source in sorted(os.listdir("data")):
        if os.path.isdir(os.path.join("data", source)):
            print("data/" + source + ":", len(os.listdir(os.path.join("data", source))), "files")
    print("data/ writable:", os.access("data", os.W_OK))

    # the pipeline's output folder
    os.makedirs("datamart", exist_ok=True)
    print("datamart/ writable:", os.access("datamart", os.W_OK))

    spark.stop()


if __name__ == "__main__":
    main()
