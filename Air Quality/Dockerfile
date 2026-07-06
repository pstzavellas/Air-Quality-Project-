# Base: Apache Spark official image (Spark 4.x με Hadoop 3.x)
FROM apache/spark:latest

# Metadata
LABEL maintainer="me25088"
LABEL description="Air Quality K-Means Clustering - Spark MLlib service"
LABEL version="1.0"

# Switch to root για installations
USER root

# Εγκατάσταση Python dependencies (numpy preinstalled)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Δημιουργία directories
RUN mkdir -p /opt/spark/scripts \
    /opt/spark/external_jars \
    /opt/spark/conf

# Αντιγραφή S3A jars (πρέπει να υπάρχουν στο context)
COPY jars/hadoop-aws-3.3.4.jar /opt/spark/external_jars/
COPY jars/aws-java-sdk-bundle-1.12.262.jar /opt/spark/external_jars/

# Αντιγραφή Spark configuration
COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf

# Αντιγραφή του K-Means script
COPY spark_kmeans.py /opt/spark/scripts/spark_kmeans.py

# Permissions
RUN chmod +x /opt/spark/scripts/spark_kmeans.py

# Switch back σε spark user
USER spark

# Working directory
WORKDIR /opt/spark

# Default command: τρέχει το K-Means job
CMD ["/opt/spark/bin/spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--conf", "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp/ivy -Divy.home=/tmp/ivy", \
     "/opt/spark/scripts/spark_kmeans.py"]