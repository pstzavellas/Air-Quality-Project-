"""
silhouette_test.py  —  ΑΥΤΟΝΟΜΟΣ έλεγχος επιλογής k (READ-ONLY)
================================================================
Διαβάζει τα ίδια snapshots από το MinIO, υπολογίζει για k=2..6:
  - silhouette score (όσο πιο κοντά στο 1, τόσο καλύτερος διαχωρισμός)
  - WSSSE / training cost (για elbow)
ΔΕΝ γράφει ΤΙΠΟΤΑ στο MinIO. Δεν πειράζει το spark_kmeans.py
ούτε τα αποτελέσματα/μοντέλα

Τρέξιμο (μία γραμμή):
  docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp/ivy -Divy.home=/tmp/ivy" \
    --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
    /opt/spark/scripts/silhouette_test.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, avg, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# --- ίδια S3A configuration με το κανονικό script (Spark 4.x compatible) ---
S3A_CONFIGS = {
    "fs.s3a.endpoint": "http://minio:9000",
    "fs.s3a.access.key": "minioadmin",
    "fs.s3a.secret.key": "minioadmin",
    "fs.s3a.path.style.access": "true",
    "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "fs.s3a.connection.ssl.enabled": "false",
    "fs.s3a.aws.credentials.provider":
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    "fs.s3a.threads.keepalivetime": "60",
    "fs.s3a.connection.establish.timeout": "5000",
    "fs.s3a.connection.timeout": "200000",
    "fs.s3a.socket.send.buffer": "8192",
    "fs.s3a.socket.recv.buffer": "8192",
    "fs.s3a.attempts.maximum": "10",
    "fs.s3a.retry.limit": "7",
    "fs.s3a.retry.interval": "500",
    "fs.s3a.retry.throttle.limit": "20",
    "fs.s3a.retry.throttle.interval": "1000",
    "fs.s3a.multipart.purge.age": "86400",
    "fs.s3a.threads.max": "10",
    "fs.s3a.max.total.tasks": "5",
}

builder = SparkSession.builder \
    .appName("AirQuality_Silhouette_Test") \
    .master("spark://spark-master:7077")
for k, v in S3A_CONFIGS.items():
    builder = builder.config(f"spark.hadoop.{k}", v)
spark = builder.getOrCreate()
hc = spark.sparkContext._jsc.hadoopConfiguration()
for k, v in S3A_CONFIGS.items():
    hc.set(k, v)

print("=" * 60)
print("Silhouette / Elbow test (READ-ONLY) — k = 2..6")
print("=" * 60)

# --- ίδια ανάγνωση & aggregation με το κανονικό script ---
df = spark.read.option("multiline", "true").json(
    "s3a://air-quality/gr_air_quality_*.json")
print(f"Snapshots: {df.count()}")

exploded = df.select(explode(col("stations")).alias("s")).select(
    col("s.measurements.pm25.value").alias("pm25"),
    col("s.measurements.pm10.value").alias("pm10"),
    col("s.measurements.no2.value").alias("no2"),
    col("s.station_id").alias("station_id"),
)
station_stats = exploded.groupBy("station_id").agg(
    avg("pm25").alias("avg_pm25"),
    avg("pm10").alias("avg_pm10"),
    avg("no2").alias("avg_no2"),
).fillna(0, subset=["avg_pm25", "avg_pm10", "avg_no2"])
print(f"Stations: {station_stats.count()}")

features_df = VectorAssembler(
    inputCols=["avg_pm25", "avg_pm10", "avg_no2"],
    outputCol="features").transform(station_stats)
features_df.cache()

evaluator = ClusteringEvaluator(
    featuresCol="features", predictionCol="prediction",
    metricName="silhouette")

print("\n  k | silhouette |   WSSSE")
print("  --+------------+----------")
results = []
for kk in range(2, 7):
    model = KMeans(featuresCol="features", k=kk, seed=42).fit(features_df)
    preds = model.transform(features_df)
    sil = evaluator.evaluate(preds)
    wssse = model.summary.trainingCost
    results.append((kk, sil, wssse))
    print(f"  {kk} |   {sil:6.4f}   | {wssse:9.2f}")

best = max(results, key=lambda r: r[1])
print(f"\n>>> Υψηλότερο silhouette: k={best[0]} (score={best[1]:.4f})")
print(">>> (Δεν γράφτηκε τίποτα στο MinIO.)")

spark.stop()