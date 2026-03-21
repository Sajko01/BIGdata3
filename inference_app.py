


from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# ==============================
# 1. SPARK SESSION
# ==============================
spark = SparkSession.builder \
    .appName("Stress Prediction Inference - Unseen Data") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================
# 2. UCITAVANJE I PODELA PODATAKA (KLJUČNI KORAK)
# ==============================
hdfs_data_path = "hdfs://namenode3:9000/user/spark/college_big.csv"

try:
    df_full = spark.read.csv(hdfs_data_path, header=True, inferSchema=True)
    
    # Koristimo ISTI split i ISTI seed kao u treningu (42)
    # train_part nam ovde ne treba, pa ga nazivamo "_" (placeholder)
    _, test_df = df_full.randomSplit([0.8, 0.2], seed=42)
    
    print(f"\n=== DATA LOADED FROM HDFS ===")
    print(f"Total rows in CSV: {df_full.count()}")
    print(f"Rows for Inference (Unseen 20%): {test_df.count()}")
    
except Exception as e:
    print(f"Greska pri ucitavanju podataka sa HDFS: {e}")
    spark.stop()
    exit()

# ==============================
# 3. LABEL (BINARIZACIJA) - NA TEST SKUPU
# ==============================
df = test_df.withColumn(
    "label",
    when(test_df["stress"] <= 2, 0).otherwise(1)
)

# ==============================
# 4. UCITAVANJE MODELA SA HDFS
# ==============================
# Putanja gde si prethodno sacuvao modele (rf_pipeline i gbt_pipeline)
hdfs_model_path = "hdfs://namenode3:9000/user/spark/models/"

print("\n=== LOADING TRAINED MODELS FROM HDFS ===")
try:
    rf_model = PipelineModel.load(hdfs_model_path + "rf_pipeline")
    gbt_model = PipelineModel.load(hdfs_model_path + "gbt_pipeline")
    print("✅ Modeli su uspesno ucitani.\n")
except Exception as e:
    print(f"Greska pri ucitavanju modela: {e}")
    spark.stop()
    exit()

# ==============================
# 5. INFERENCE (PREDIKCIJA NA NEVIDJENIM PODACIMA)
# ==============================
print("=== IZVRSAVAM PREDIKCIJE NA TEST SETU... ===")

# Koristimo .transform() nad nevidjenim test podacima
rf_results = rf_model.transform(df)
gbt_results = gbt_model.transform(df)

# Prikazujemo prvih 10 rezultata da vidimo sta je model "pogodio"
print("\n--- RANDOM FOREST REZULTATI (Primer) ---")
rf_results.select("phq4_score", "label", "prediction", "probability").show(10)

# ==============================
# 6. EVALUACIJA PERFORMANSI
# ==============================
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

rf_acc = evaluator.evaluate(rf_results)
gbt_acc = evaluator.evaluate(gbt_results)

print("=" * 50)
print(f"FINALNA TACNOST NA NEVIDJENIM PODACIMA (Accuracy):")
print(f"Random Forest: {rf_acc:.4f}")
print(f"GBT Classifier: {gbt_acc:.4f}")
print("=" * 50)

# ==============================
# 7. CUVANJE REZULTATA PREDIKCIJE
# ==============================
output_path = "hdfs://namenode3:9000/user/spark/results/final_inference/"

# Cuvamo rezultate u Parquet formatu za dalju analizu/vizuelizaciju
rf_results.select("label", "prediction", "probability") \
    .write.mode("overwrite") \
    .parquet(output_path + "rf_final")

gbt_results.select("label", "prediction", "probability") \
    .write.mode("overwrite") \
    .parquet(output_path + "gbt_final")

print(f"\n✅ Rezultati inferencije sacuvani na: {output_path}")

spark.stop()