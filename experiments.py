

from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. SPARK SESSION (LOKALNI MOD)
# ==============================
# Koristimo .master("local[*]") da Spark koristi sve procesore tvog kompjutera
spark = SparkSession.builder \
    .appName("ML Parameter Experiments Local") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================
# 2. LOAD DATA (LOKALNA PUTANJA)
# ==============================
# Umesto HDFS-a, koristimo tvoj D: disk
csv_path = "D:/02BIGDATA/college_big.csv"

if not os.path.exists(csv_path):
    print(f"Greška: Fajl nije pronađen na {csv_path}")
    spark.stop()
    exit()

df = spark.read.csv(csv_path, header=True, inferSchema=True)

# DODAJ OVU LINIJU:
df = df.sample(fraction=0.3, seed=42) # Uzmi samo 30% podataka za brzi test

df = df.withColumn(
    "label",
    when(df["stress"] <= 2, 0).otherwise(1)
)

# ==============================
# 3. SPLIT
# ==============================
train, test = df.randomSplit([0.8, 0.2], seed=42)

train.cache()
test.cache()

# ==============================
# 4. PIPELINE KOMPONENTE
# ==============================
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
race_indexer = StringIndexer(inputCol="race", outputCol="race_index", handleInvalid="keep")

assembler = VectorAssembler(
    inputCols=[
        "phq4_score", "social_level", "sleep_duration",
        "daily_steps", "covid_total",
        "gender_index", "race_index"
    ],
    outputCol="features"
)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# ==============================
# 5. PARAMETRI
# ==============================
numTrees_list = [10, 50]
maxDepth_list = [3, 6]

results = []

# ==============================
# 6. RANDOM FOREST EXPERIMENTI
# ==============================
print("\n=== POKREĆEM RF EXPERIMENTE ===")

for nt in numTrees_list:
    for md in maxDepth_list:
        print(f"Treniram RF: Stabala={nt}, Dubina={md}...")

        rf = RandomForestClassifier(
            featuresCol="scaledFeatures",
            labelCol="label",
            numTrees=nt,
            maxDepth=md
        )

        pipeline = Pipeline(stages=[
            gender_indexer, race_indexer, assembler, scaler, rf
        ])

        start = time.time()
        model = pipeline.fit(train)
        train_time = time.time() - start

        predictions = model.transform(test)
        acc = evaluator.evaluate(predictions)

        results.append({
            "model": "RF",
            "params": f"Trees:{nt}, Depth:{md}",
            "accuracy": acc,
            "train_time": train_time
        })

# ==============================
# 7. GBT EXPERIMENTI
# ==============================
print("\n=== POKREĆEM GBT EXPERIMENTE ===")

for md in maxDepth_list:
    for it in [10, 25]:
        print(f"Treniram GBT: Dubina={md}, Iteracija={it}...")

        gbt = GBTClassifier(
            featuresCol="scaledFeatures",
            labelCol="label",
            maxDepth=md,
            maxIter=it
        )

        pipeline = Pipeline(stages=[
            gender_indexer, race_indexer, assembler, scaler, gbt
        ])

        start = time.time()
        model = pipeline.fit(train)
        train_time = time.time() - start

        predictions = model.transform(test)
        acc = evaluator.evaluate(predictions)

        results.append({
            "model": "GBT",
            "params": f"Depth:{md}, Iter:{it}",
            "accuracy": acc,
            "train_time": train_time
        })

# ==============================
# 8. REZULTATI U DATAFRAME
# ==============================
results_df = pd.DataFrame(results)

print("\n=== TABELA REZULTATA ===")
print(results_df)

# ==============================
# 9. GRAFIKONI
# ==============================
# Accuracy grafik
plt.figure(figsize=(12, 5))
for model_type in results_df['model'].unique():
    subset = results_df[results_df['model'] == model_type]
    plt.plot(subset['params'], subset['accuracy'], marker='o', label=model_type)

plt.title("Poređenje tačnosti (Accuracy) po eksperimentima")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Vreme treniranja grafik
plt.figure(figsize=(12, 5))
for model_type in results_df['model'].unique():
    subset = results_df[results_df['model'] == model_type]
    plt.plot(subset['params'], subset['train_time'], marker='s', label=model_type)

plt.title("Vreme treniranja modela")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Vreme (sekunde)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==============================
# 10. SACUVAJ REZULTATE
# ==============================
output_dir = "D:/02BIGDATA/results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results_df.to_csv(f"{output_dir}/experiment_results.csv", index=False)
print(f"\n✅ REZULTATI SAČUVANI: {output_dir}/experiment_results.csv")

spark.stop()