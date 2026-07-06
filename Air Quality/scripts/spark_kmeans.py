"""
Air Quality K-Means Clustering of Greek Air Quality Stations
============================================================
Διαβάζει όλα τα ιστορικά JSON snapshots από MinIO,
υπολογίζει στατιστικά ανά σταθμό (μέσος όρος PM2.5/PM10/NO2),
και ομαδοποιεί τους σταθμούς της Ελλάδας σε 4 κατηγορίες
με βάση το προφίλ ρύπανσής τους.

Αποθηκεύει στο MinIO:
- s3a://air-quality/results/station_clusters/  (αποτελέσματα ανά σταθμό)
- s3a://air-quality/models/kmeans_stations/    (trained K-Means model)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, avg, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# ============================================================
# S3A Configuration - Spark 4.x + hadoop-aws 3.3.4 compatibility
# ============================================================
# Όλες οι τιμές integers (το Spark 4.x στέλνει "Xs" format
# που σπάει στο hadoop-aws 3.3.4 που περιμένει pure integers)
S3A_CONFIGS = {
    # Connection settings
    "fs.s3a.endpoint": "http://minio:9000",
    "fs.s3a.access.key": "minioadmin",
    "fs.s3a.secret.key": "minioadmin",
    "fs.s3a.path.style.access": "true",
    "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "fs.s3a.connection.ssl.enabled": "false",
    # Credentials provider (SDK v1, που έχει το hadoop-aws 3.3.4)
    "fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    # Timeout/buffer overrides - integer values
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

# ============================================================
# 1. Initialize Spark Session
# ============================================================
builder = SparkSession.builder \
    .appName("AirQuality_KMeans_Stations") \
    .master("spark://spark-master:7077")

for key, value in S3A_CONFIGS.items():
    builder = builder.config(f"spark.hadoop.{key}", value)

spark = builder.getOrCreate()

# Double safety - override Hadoop config object directly
hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
for key, value in S3A_CONFIGS.items():
    hadoop_conf.set(key, value)

print("=" * 70)
print("Air Quality K-Means Clustering of Greek Air Quality Stations")
print("=" * 70)

# ============================================================
# 2. Read all JSON snapshots from MinIO
# ============================================================
path = "s3a://air-quality/gr_air_quality_*.json"
print(f"\n[1/6] Reading snapshots from {path}")

df = spark.read.option("multiline", "true").json(path)
snapshot_count = df.count()
print(f"  Found {snapshot_count} snapshot(s)")

if snapshot_count == 0:
    print("  ERROR: No snapshots found. Exiting.")
    spark.stop()
    exit(1)

# ============================================================
# 3. Explode stations array - one row per station per snapshot
# ============================================================
print("\n[2/6] Exploding stations from each snapshot")
exploded = df.select(
    col("ingestion_timestamp"),
    explode(col("stations")).alias("s")
).select(
    col("ingestion_timestamp"),
    col("s.station_id").alias("station_id"),
    col("s.station_name").alias("station_name"),
    col("s.locality").alias("locality"),
    col("s.coordinates.latitude").alias("latitude"),
    col("s.coordinates.longitude").alias("longitude"),
    col("s.measurements.pm25.value").alias("pm25"),
    col("s.measurements.pm10.value").alias("pm10"),
    col("s.measurements.no2.value").alias("no2")
)
print(f"  Total measurements: {exploded.count()}")

# ============================================================
# 4. Aggregate per station - average across all snapshots
# ============================================================
print("\n[3/6] Computing per-station statistics (averages)")
station_stats = exploded.groupBy(
        "station_id", "station_name", "locality", "latitude", "longitude"
    ).agg(
        avg("pm25").alias("avg_pm25"),
        avg("pm10").alias("avg_pm10"),
        avg("no2").alias("avg_no2"),
        count("*").alias("snapshot_count")
    )

# Fill nulls with 0 (σταθμοί που δεν έχουν κάποιον αισθητήρα)
station_stats = station_stats.fillna(0, subset=["avg_pm25", "avg_pm10", "avg_no2"])
station_count = station_stats.count()
print(f"  Stations: {station_count}")

# ============================================================
# 5. Train K-Means model
# ============================================================
print("\n[4/6] Training K-Means model (k=4)")
vec_assembler = VectorAssembler(
    inputCols=["avg_pm25", "avg_pm10", "avg_no2"],
    outputCol="features"
)
features_df = vec_assembler.transform(station_stats)

kmeans = KMeans(featuresCol="features", k=4, seed=42)
model = kmeans.fit(features_df)

# ============================================================
# 6. Show cluster centers and predictions
# ============================================================
print("\n[5/6] Cluster Centers (PM2.5, PM10, NO2):")
centers = model.clusterCenters()
for i, c in enumerate(centers):
    print(f"  Cluster {i}: PM2.5={c[0]:6.2f}, PM10={c[1]:6.2f}, NO2={c[2]:6.2f}")

# Apply predictions
predictions = model.transform(features_df).select(
    "station_id", "station_name", "locality",
    "latitude", "longitude",
    "avg_pm25", "avg_pm10", "avg_no2",
    "snapshot_count", "prediction"
)

print("\n  Stations per cluster:")
predictions.groupBy("prediction").count().orderBy("prediction").show()

print("\n  Sample of station classifications:")
predictions.orderBy("prediction", "station_name").show(20, truncate=False)

# ============================================================
# 7. Save results to MinIO
# ============================================================
print("\n[6/6] Saving results to MinIO")

# Save predictions as a single JSON file
predictions.coalesce(1).write.mode("overwrite") \
    .json("s3a://air-quality/results/station_clusters")
print("  ✓ Saved: s3a://air-quality/results/station_clusters/")

# Save trained model
model.write().overwrite().save("s3a://air-quality/models/kmeans_stations")
print("  ✓ Saved: s3a://air-quality/models/kmeans_stations/")

# Save centroids ως απλό JSON για εύκολη ανάγνωση από Node-RED
import json
from datetime import datetime, timezone

centroids_data = {
    "model_name": "kmeans_air_quality_stations",
    "k": 4,
    "features": ["pm25", "pm10", "no2"],
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "num_stations": station_count,
    "num_snapshots": snapshot_count,
    "centroids": [
        {
            "cluster_id": i,
            "pm25": float(c[0]),
            "pm10": float(c[1]),
            "no2": float(c[2])
        }
        for i, c in enumerate(centers)
    ]
}

# Write to local first, then upload using Spark
centroids_json = json.dumps(centroids_data, indent=2)
print("\n  Centroids JSON:")
print(centroids_json)

# Save to MinIO via Spark
spark.createDataFrame([(centroids_json,)], ["json"]) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .text("s3a://air-quality/models/centroids_raw")

# Επίσης γράφουμε ένα structured JSON για εύκολη ανάγνωση
centroids_df = spark.createDataFrame([
    (i, float(c[0]), float(c[1]), float(c[2]))
    for i, c in enumerate(centers)
], ["cluster_id", "pm25", "pm10", "no2"])

centroids_df.coalesce(1).write.mode("overwrite") \
    .json("s3a://air-quality/models/centroids")
print("  ✓ Saved: s3a://air-quality/models/centroids/")
# ============================================================
# 9. Save centroids με σταθερό όνομα για εύκολη πρόσβαση από Node-RED
# ============================================================
print("\n[8/8] Writing fixed-name centroids file for Node-RED")
output_path = spark._jvm.org.apache.hadoop.fs.Path(
    "s3a://air-quality/models/latest_centroids.json"
)
hadoop_fs = output_path.getFileSystem(spark._jsc.hadoopConfiguration())
output_stream = hadoop_fs.create(output_path, True)
output_stream.write(centroids_json.encode('utf-8'))
output_stream.close()
print("  ✓ Saved: s3a://air-quality/models/latest_centroids.json")
print("\n" + "=" * 70)
print("Done!")
print("=" * 70)

spark.stop()