FROM apache/airflow:3.3.2-python3.12

# Java for Spark (installed as root, like Lab 4's Dockerfile)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# one JAVA_HOME path that works on both Apple silicon (arm64) and Intel/AMD (amd64)
RUN ln -s "/usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture)" /usr/lib/jvm/java-17
ENV JAVA_HOME=/usr/lib/jvm/java-17

# Python packages, installed as the airflow user (Airflow's custom-image guide)
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt

