from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Pokrećemo minimalni Spark session (ne treba nam mnogo RAM-a za ovo)
spark = SparkSession.builder.appName("Quick Importance Extraction").getOrCreate()

# 1. Učitavamo već istrenirane modele sa HDFS-a
print("Učitavam modele...")
rf_model = PipelineModel.load("hdfs://namenode3:9000/user/spark/models/rf_pipeline")
gbt_model = PipelineModel.load("hdfs://namenode3:9000/user/spark/models/gbt_pipeline")

# 2. Definišemo nazive kolona (identično kao u treningu)
feature_cols = ["phq4_score", "social_level", "sleep_duration", "daily_steps", "covid_total", "gender_index", "race_index"]

def save_only_importance(model_pipeline, model_name):
    # Uzimamo klasifikator iz poslednjeg stage-a
    trained_classifier = model_pipeline.stages[-1]
    importances = trained_classifier.featureImportances.toArray()
    
    # Mapiranje u listu
    fi_data = [(feature_cols[i], float(importances[i])) for i in range(len(feature_cols))]
    
    # Pravimo DF i čuvamo ga
    fi_df = spark.createDataFrame(fi_data, ["Feature", "Importance"])
    path = f"hdfs://namenode3:9000/user/spark/results/final_inference/importance_{model_name}"
    
    fi_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
    print(f"✅ Uspešno izvučeno za {model_name}")

# 3. Izvršavamo ekstrakciju
save_only_importance(rf_model, "rf")
save_only_importance(gbt_model, "gbt")  # Ovde je falilo "_only"

spark.stop()