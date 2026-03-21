

from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ==============================
# 1. SPARK SESSION
# ==============================
spark = SparkSession.builder \
    .appName("Stress Prediction Training") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

# ==============================
# 2. UCITAVANJE PODATAKA
# ==============================
df = spark.read.csv("hdfs://namenode3:9000/college_big.csv", header=True, inferSchema=True)
#df = spark.read.csv("college_big.csv", header=True, inferSchema=True)

# ==============================
# 3. LABEL (BINARIZACIJA) - Ovo mora pre Pipeline-a
# ==============================
df = df.withColumn(
    "label",
    when(df["stress"] <= 2, 0).otherwise(1)
)

# ==============================
# 4. DEFINISANJE PIPELINE FAZA
# ==============================

# Indexeri za kategorijalne podatke (handleInvalid="keep" je sigurnije za nove podatke)
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
race_indexer = StringIndexer(inputCol="race", outputCol="race_index", handleInvalid="keep")

# Sastavljanje vektora featura
assembler = VectorAssembler(
    inputCols=[
        "phq4_score",
        "social_level",
        "sleep_duration",
        "daily_steps",
        "covid_total",
        "gender_index",
        "race_index"
    ],
    outputCol="features"
)

# Skaliranje
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

# Modeli
rf = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    numTrees=50,
    maxDepth=5
)

gbt = GBTClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=20,
    maxDepth=5
)

# ==============================
# 5. TRAIN / TEST SPLIT
# ==============================
train, test = df.randomSplit([0.8, 0.2], seed=42)

# ==============================
# 6. TRENIRANJE KROZ PIPELINE
# ==============================

# Kreiramo dva odvojena pipeline-a jer imamo dva različita modela na kraju
pipeline_rf = Pipeline(stages=[gender_indexer, race_indexer, assembler, scaler, rf])
pipeline_gbt = Pipeline(stages=[gender_indexer, race_indexer, assembler, scaler, gbt])

print("Trening u toku...")
rf_model = pipeline_rf.fit(train)
gbt_model = pipeline_gbt.fit(train)

# ==============================
# 7. EVALUACIJA (Koristeći transform na ceo pipeline)
# ==============================
rf_predictions = rf_model.transform(test)
gbt_predictions = gbt_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

print(f"Random Forest Accuracy: {evaluator.evaluate(rf_predictions)}")
print(f"GBT Accuracy: {evaluator.evaluate(gbt_predictions)}")

# ==============================
# 8. ČUVANJE CELIH PIPELINE MODELA NA HDFS
# ==============================
# VEOMA BITNO: Snimamo ceo rf_model (PipelineModel), ne samo rf (Classifier)
# rf_model.write().overwrite().save("hdfs://namenode3:9000/models/rf_pipeline")
# gbt_model.write().overwrite().save("hdfs://namenode3:9000/models/gbt_pipeline")

# Putanja koja obično uvek radi za korisnika spark
rf_model.write().overwrite().save("hdfs://namenode3:9000/user/spark/models/rf_pipeline")
gbt_model.write().overwrite().save("hdfs://namenode3:9000/user/spark/models/gbt_pipeline")

print("=== PIPELINE MODELI USPESNO SACUVANI NA HDFS ===")

spark.stop()