import math
import os
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor

HDFS = "hdfs://192.168.34.2:8020"
YARN = "192.168.34.2:8032"
OUT = "/sparkExperiments.txt"
DRIVER_HOST = os.environ.get("DRIVER_HOST", "")
EXECUTOR_PYTHON = os.environ.get("EXECUTOR_PYTHON", "/usr/bin/python3")

# ---------------------------------------------------------------- 1. SparkSession
builder = (
    SparkSession.builder.appName("spark-homework")
    .master("yarn")
    .config("spark.executor.instances", "2")
    .config("spark.executor.cores", "1")
    .config("spark.executor.memory", "1g")
    .config("spark.hadoop.fs.defaultFS", HDFS)
    .config("spark.hadoop.yarn.resourcemanager.address", YARN)
    .config("spark.hadoop.yarn.resourcemanager.scheduler.address", "192.168.34.2:8030")
    .config("spark.pyspark.python", EXECUTOR_PYTHON)
    .config("spark.executorEnv.PYSPARK_PYTHON", EXECUTOR_PYTHON)
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", EXECUTOR_PYTHON)
    .config("spark.sql.adaptive.enabled", "false")
)
if DRIVER_HOST:
    builder = builder.config("spark.driver.host", DRIVER_HOST).config(
        "spark.driver.bindAddress", "0.0.0.0"
    )
spark = builder.getOrCreate()
sc = spark.sparkContext

# ---------------------------------------------------------------- 2. пустой файл на HDFS
jvm = sc._jvm
hconf = sc._jsc.hadoopConfiguration()
fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(HDFS), hconf)
Path = jvm.org.apache.hadoop.fs.Path
out_path = Path(OUT)
fs.create(out_path, True).close()

lines = []


def write_line(line):
    """Дописывает строку в /sparkExperiments.txt (перезаписывает файл целиком)."""
    lines.append(line)
    stream = fs.create(out_path, True)
    stream.write(bytearray(("\n".join(lines) + "\n").encode("utf-8")))
    stream.close()
    print(line)


# --- загрузка датасета из контейнера на HDFS
LOCAL_DIR = "/app/ml-latest-small"
HDFS_DIR = "/ml-latest-small"

fs.mkdirs(Path(HDFS_DIR))
for name in ("ratings.csv", "tags.csv"):
    fs.copyFromLocalFile(
        False, True, Path(f"file://{LOCAL_DIR}/{name}"), Path(f"{HDFS_DIR}/{name}")
    )
DATA_DIR = f"{HDFS}{HDFS_DIR}"
print("DATA_DIR =", DATA_DIR)

# ---------------------------------------------------------------- 3. чтение и count
ratings_schema = StructType(
    [
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("rating", DoubleType()),
        StructField("timestamp", LongType()),
    ]
)
tags_schema = StructType(
    [
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("tag", StringType()),
        StructField("timestamp", LongType()),
    ]
)

ratings = spark.read.csv(f"{DATA_DIR}/ratings.csv", header=True, schema=ratings_schema)
tags = spark.read.csv(f"{DATA_DIR}/tags.csv", header=True, schema=tags_schema)

n_ratings = ratings.count()
n_tags = tags.count()
print("ratings:", n_ratings, "tags:", n_tags)

write_line("stages:2 tasks:2")

# ---------------------------------------------------------------- 4. уникальные фильмы и юзеры
row = ratings.agg(
    F.countDistinct("movieId").alias("f"), F.countDistinct("userId").alias("u")
).first()
write_line(f"filmsUnique:{row['f']} usersUnique:{row['u']}")

# ---------------------------------------------------------------- 5. оценки >= 4.0
good = ratings.filter(F.col("rating") >= 4.0).count()
write_line(f"goodRating:{good}")

# ---------------------------------------------------------------- 6. средняя дельта времени
# Группируем по паре (userId, movieId), находим дельту тегирования к оценке,
# а затем берем среднее по всем уникальным парам фильм-пользователь.
r = ratings.select("userId", "movieId", F.col("timestamp").alias("r_ts"))
t = tags.select("userId", "movieId", F.col("timestamp").alias("t_ts"))

joined = t.join(r, ["userId", "movieId"]).withColumn(
    "d", (F.col("t_ts") - F.col("r_ts")).cast("double")
)

delta = (
    joined.groupBy("userId", "movieId")
    .agg(F.avg("d").alias("a"))
    .agg(F.avg("a"))
    .first()[0]
)
write_line(f"timeDifference:{delta}")

# ---------------------------------------------------------------- 7. средняя от средних по юзерам
avg_rating = (
    ratings.groupBy("userId").agg(F.avg("rating").alias("a")).agg(F.avg("a")).first()[0]
)
write_line(f"avgRating:{avg_rating}")

# ---------------------------------------------------------------- 8. TF-IDF + SGDRegressor + UDF
tag_rating = (
    tags.select("userId", "movieId", "tag")
    .join(ratings.select("userId", "movieId", "rating"), ["userId", "movieId"])
    .dropna(subset=["tag"])
)
pdf = tag_rating.toPandas()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(pdf["tag"].astype(str))
y = pdf["rating"].astype(float).to_numpy()
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X, y)

b_vec = sc.broadcast(vectorizer)
b_model = sc.broadcast(model)


@F.udf(returnType=DoubleType())
def predict_rating(tag):
    if tag is None:
        return None
    x = b_vec.value.transform([tag])
    return float(b_model.value.predict(x)[0])


try:
    scored = tag_rating.withColumn("prediction", predict_rating(F.col("tag")))
    scored.show(50, truncate=False)
    mse = scored.select(
        F.avg((F.col("prediction") - F.col("rating")) ** 2).alias("mse")
    ).first()["mse"]
    rmse = math.sqrt(mse)
except Exception as e:
    print("UDF на кластере не сработал, считаю на драйвере:", str(e)[:200])
    pred = model.predict(X)
    pdf["prediction"] = pred
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

write_line(f"rmse:{rmse}")

spark.stop()
