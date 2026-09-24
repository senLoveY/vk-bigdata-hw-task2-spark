FROM --platform=linux/amd64 python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python3
ENV HADOOP_USER_NAME=root
ENV HADOOP_CONF_DIR=/etc/hadoop/conf
ENV YARN_CONF_DIR=/etc/hadoop/conf

RUN mkdir -p /etc/hadoop/conf \
    && printf '%s\n' \
      '<?xml version="1.0"?>' \
      '<configuration>' \
      '  <property><name>fs.defaultFS</name><value>hdfs://192.168.34.2:8020</value></property>' \
      '</configuration>' > /etc/hadoop/conf/core-site.xml \
    && printf '%s\n' \
      '<?xml version="1.0"?>' \
      '<configuration>' \
      '  <property><name>yarn.resourcemanager.hostname</name><value>192.168.34.2</value></property>' \
      '  <property><name>yarn.resourcemanager.address</name><value>192.168.34.2:8032</value></property>' \
      '  <property><name>yarn.resourcemanager.scheduler.address</name><value>192.168.34.2:8030</value></property>' \
      '</configuration>' > /etc/hadoop/conf/yarn-site.xml

RUN pip install --no-cache-dir pyspark==3.5.1 pandas numpy scikit-learn

WORKDIR /app
COPY ml-latest-small /app/ml-latest-small
COPY spark_app.py /app/spark_app.py

CMD ["sh", "-c", "export DRIVER_HOST=${DRIVER_HOST:-$(hostname -i | awk '{print $1}')}; python /app/spark_app.py"]