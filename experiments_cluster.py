from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import pandas as pd
import os

# ==============================
# 1. SPARK SESSION (DOCKER KLASTER MOD)
# ==============================
# Master se ne definiše ovde (local[*]), već preko spark-submit komande
spark = SparkSession.builder \
    .appName("ML Cluster Experiments - Fast Mode") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================
# 2. LOAD DATA (HDFS PUTANJA)
# ==============================
# Docker kontejner čita podatke direktno sa HDFS-a
hdfs_path = "hdfs://namenode3:9000/user/spark/college_big.csv"

print(f"\n>>> Učitavam podatke sa HDFS: {hdfs_path}")
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# SAMPLING: Uzimamo 30% podataka radi brzine (identično kao u lokalnom testu)
df = df.sample(fraction=0.3, seed=42)

df = df.withColumn(
    "label",
    when(df["stress"] <= 2, 0).otherwise(1)
)

# ==============================
# 3. SPLIT I CACHE
# ==============================
train, test = df.randomSplit([0.8, 0.2], seed=42)

# CACHE je ključan za brzinu u petljama na klasteru
train.cache()
test.cache()

print(f"Broj redova za trening: {train.count()}")

# ==============================
# 4. PIPELINE KOMPONENTE
# ==============================
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
race_indexer = StringIndexer(inputCol="race", outputCol="race_index", handleInvalid="keep")

assembler = VectorAssembler(
    inputCols=["phq4_score", "social_level", "sleep_duration", "daily_steps", "covid_total", "gender_index", "race_index"],
    outputCol="features"
)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# ==============================
# 5. PARAMETRI (Optimizovani za brzinu)
# ==============================
numTrees_list = [10, 50]
maxDepth_list = [3, 6]
results = []

# ==============================
# 6. RANDOM FOREST EXPERIMENTI
# ==============================
print("\n=== POKREĆEM RF EXPERIMENTE NA KLASTERU ===")

for nt in numTrees_list:
    for md in maxDepth_list:
        print(f"RF -> Trees:{nt}, Depth:{md}")

        rf = RandomForestClassifier(
            featuresCol="scaledFeatures",
            labelCol="label",
            numTrees=nt,
            maxDepth=md
        )

        pipeline = Pipeline(stages=[gender_indexer, race_indexer, assembler, scaler, rf])

        start = time.time()
        model = pipeline.fit(train)
        train_time = time.time() - start

        predictions = model.transform(test)
        acc = evaluator.evaluate(predictions)

        results.append({
            "model": "RF",
            "params": f"T:{nt}, D:{md}",
            "accuracy": acc,
            "train_time": train_time
        })

# ==============================
# 7. GBT EXPERIMENTI
# ==============================
print("\n=== POKREĆEM GBT EXPERIMENTE NA KLASTERU ===")

for md in maxDepth_list:
    for it in [10, 25]:
        print(f"GBT -> Depth:{md}, Iter:{it}")

        gbt = GBTClassifier(
            featuresCol="scaledFeatures",
            labelCol="label",
            maxDepth=md,
            maxIter=it
        )

        pipeline = Pipeline(stages=[gender_indexer, race_indexer, assembler, scaler, gbt])

        start = time.time()
        model = pipeline.fit(train)
        train_time = time.time() - start

        predictions = model.transform(test)
        acc = evaluator.evaluate(predictions)

        results.append({
            "model": "GBT",
            "params": f"D:{md}, I:{it}",
            "accuracy": acc,
            "train_time": train_time
        })

# ==============================
# 8. REZULTATI I ČUVANJE
# ==============================
results_df = pd.DataFrame(results)

print("\n=== FINALNA TABELA REZULTATA ===")
print(results_df)

# Čuvamo CSV unutar Docker master kontejnera
output_path = "/opt/spark/work-dir/experiment_results_cluster.csv"
results_df.to_csv(output_path, index=False)

print(f"\n✅ REZULTATI SAČUVANI U KONTEJNERU: {output_path}")

spark.stop()