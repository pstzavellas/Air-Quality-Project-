Real-Time Air-Quality Analytics — Lambda Architecture

A containerised big-data pipeline that ingests live air-quality sensor data, stores it in an object store, processes it in a distributed Spark cluster, and segments pollution profiles with K-Means clustering. Built as a full Lambda Architecture (batch + speed layers) running entirely on Docker.


Stack: Apache Spark · Node-RED · MinIO · Docker · Python (PySpark) · K-Means




Architecture

                 ┌──────────────┐
  Air-quality    │   Node-RED   │   ingestion / speed layer
  sensor API ───▶│   (flows)    │──────────────┐
                 └──────────────┘              │
                                               ▼
                                     ┌────────────────────┐
                                     │       MinIO        │  object storage
                                     │  (S3-compatible)   │  (raw + curated)
                                     └────────────────────┘
                                               │
                                               ▼
                 ┌───────────────────────────────────────────────┐
                 │            Apache Spark cluster                │  batch layer
                 │   1 master  +  2 workers  (standalone mode)    │
                 │   PySpark job: clean → feature → K-Means       │
                 └───────────────────────────────────────────────┘
                                               │
                                               ▼
                                   clustered pollution profiles


Speed layer — Node-RED polls the air-quality source and lands records into MinIO continuously.
Batch layer — a Spark job reads from MinIO, cleans and vectorises the data, and trains a K-Means model to group measurements into pollution profiles.
Serving — results are written back to MinIO for downstream consumption.



Tech Stack

LayerTechnologyRoleIngestionNode-REDPulls sensor data and writes to object storageStorageMinIOS3-compatible object store (raw & curated zones)ProcessingApache Spark (1 master + 2 workers)Distributed cleaning, feature engineering, clusteringMLSpark MLlib — K-MeansUnsupervised segmentation of pollution profilesOrchestrationDocker ComposeSingle-command, reproducible multi-service cluster


Repository Contents

FileDescriptiondocker-compose.ymlDefines the full cluster: Node-RED, MinIO, Spark master + 2 workersDockerfileCustom Spark image buildflows.jsonNode-RED ingestion flow (the speed layer)spark_kmeans.pyMain PySpark job: load → clean → vectorise → K-Meanssilhouette_test.pyCluster-quality analysis to select the number of clusters kspark-defaults.confSpark configuration (incl. MinIO / S3A connector settings)


Note on structure: the source files are kept flat in this repository for readability. In the running environment the scripts live under sparkdir/scripts/ and the configuration under sparkdir/conf/, as referenced by the volume mounts in docker-compose.yml.




Running the Cluster

bash# 1. Build and start every service
docker-compose up --build

# 2. Open the web UIs
#    Node-RED    → http://localhost:1880
#    MinIO       → http://localhost:9001   (default dev creds: minioadmin / minioadmin)
#    Spark master→ http://localhost:8080

# 3. Submit the clustering job to the Spark master
docker exec spark-master /opt/spark/bin/spark-submit \
  /opt/spark/scripts/spark_kmeans.py


MinIO here uses the built-in development credentials (minioadmin). In a production deployment these would be injected as environment variables / secrets rather than left at their defaults.




Results & Methodology

The K-Means model was run with k = 4 to produce interpretable pollution profiles for the report.

To justify the choice of k honestly, a separate silhouette analysis (silhouette_test.py) was run across candidate values. It showed that k = 3 was the statistical optimum (silhouette ≈ 0.76), noticeably higher than k = 4 (≈ 0.49). This trade-off — a slightly lower silhouette score in exchange for a more granular, more interpretable segmentation — is documented explicitly rather than hidden, and the discrepancy is discussed in the accompanying report.

This is a deliberate demonstration that model selection is a judgement call, not just an argmax: the "best" score and the most useful segmentation are not always the same, and the reasoning behind the decision matters.


Known Limitations & Future Work


Missing values are currently handled with a simple fillna(0). This is a documented simplification; median/interpolation imputation would be more faithful for sensor gaps.
No automated retraining. The K-Means model is trained on demand rather than on a schedule; wiring periodic retraining into the Node-RED flow (true speed-layer feedback) is the natural next step.
Serving layer is limited to writing results back to MinIO; a dedicated query/visualisation endpoint would complete the Lambda picture.



Why This Project

It exercises an end-to-end big-data workflow — distributed processing, object storage, containerised orchestration, and unsupervised ML — the same building blocks used to organise, analyse and visualise large operational datasets (e.g. vehicle-fleet, production or sensor data) in industry.


Academic project — MSc in Information Systems, University of Piraeus.
