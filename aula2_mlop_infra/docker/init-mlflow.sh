#!/bin/bash
# Starts the MLflow tracking server.

# Create the directories inside the mounted volumes if they do not exist yet, and make
# them writable whatever the host UID happens to be.
#
# TEACHING SHORTCUT: chmod 777 is not something you would ever ship. In production the
# container gets a dedicated non-root user and the volumes are created with matching
# ownership up front. We take the shortcut here so that a class of laptops with a dozen
# different user ids all work without configuration.
mkdir -p /mlflow/db /artifacts
chmod -R 777 /mlflow /artifacts

# --allowed-hosts is required since MLflow added DNS-rebinding protection: the server
# rejects any request whose Host header it does not recognise. Its default covers
# localhost and private IPs, but not Docker service names — so without this line the
# notebook's calls to http://mlflow:5000 come back as
# "Invalid Host header - possible DNS rebinding attack detected".
#
#   mlflow:5000     how the jupyter and api containers reach it
#   localhost:5001  how your browser reaches it (see the port mapping in docker-compose.yml)
exec mlflow server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --artifacts-destination /artifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "mlflow:5000,mlflow,localhost:5001,localhost:5000,localhost,127.0.0.1:5001,127.0.0.1:5000,127.0.0.1"
