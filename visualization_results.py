import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Postavljanje stila za lepše grafike
plt.style.use('ggplot')
sns.set_palette("viridis")

# ==============================
# 1. PARAMETRI I PUTANJE
# ==============================
# PROVERI: Da li si uradio 'hdfs dfs -get' u ovaj folder?
# rf_path = "D:/02BIGDATA/results/results_rf.parquet"
# gbt_path = "D:/02BIGDATA/results/results_gbt.parquet"
rf_path = "D:/02BIGDATA/results/rf_predictions"
gbt_path = "D:/02BIGDATA/results/gbt_predictions"

def load_and_clean(path):
    try:
        df = pd.read_parquet(path)
        # Spark 'Probability' kolona je često rečnik {'type': 1, 'values': [0.1, 0.9]}
        # Izvlačimo samo verovatnoću za klasu 1 (High Stress)
        if 'probability' in df.columns:
            df['prob_1'] = df['probability'].apply(
                lambda x: x['values'][1] if isinstance(x, dict) else (x[1] if hasattr(x, '__getitem__') else x)
            )
        return df
    except Exception as e:
        print(f"Greška pri učitavanju {path}: {e}")
        return None

rf_df = load_and_clean(rf_path)
gbt_df = load_and_clean(gbt_path)

if rf_df is not None and gbt_df is not None:
    print("✅ Podaci uspešno učitani!")
    
    # ==============================
    # 2. POREĐENJE TAČNOSTI (Accuracy)
    # ==============================
    rf_acc = (rf_df['label'] == rf_df['prediction']).mean()
    gbt_acc = (gbt_df['label'] == gbt_df['prediction']).mean()

    plt.figure(figsize=(10, 6))
    models = ['Random Forest', 'Gradient Boosted Trees']
    accuracies = [rf_acc, gbt_acc]
    
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c'])
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy Score')
    plt.title('Poređenje preciznosti modela na test setu')
    
    # Dodavanje procenata iznad stubića
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.show()

    # ==============================
    # 3. CONFUSION MATRICE (Uporedo)
    # ==============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    labels = ['Low Stress', 'High Stress']

    # RF Matrix
    cm_rf = confusion_matrix(rf_df['label'], rf_df['prediction'])
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=labels)
    disp_rf.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title("Confusion Matrix: Random Forest")

    # GBT Matrix
    cm_gbt = confusion_matrix(gbt_df['label'], gbt_df['prediction'])
    disp_gbt = ConfusionMatrixDisplay(confusion_matrix=cm_gbt, display_labels=labels)
    disp_gbt.plot(ax=ax2, cmap='Reds', colorbar=False)
    ax2.set_title("Confusion Matrix: GBT")

    plt.tight_layout()
    plt.show()

    # ==============================
    # 4. DISTRIBUCIJA VEROVATNOĆE (Sigurnost modela)
    # ==============================
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(rf_df['prob_1'], label='Random Forest', fill=True, alpha=0.5)
    sns.kdeplot(gbt_df['prob_1'], label='GBT', fill=True, alpha=0.5)
    plt.axvline(0.5, color='black', linestyle='--', label='Granica odlučivanja')
    plt.title('Distribucija verovatnoće za High Stress (Klasa 1)')
    plt.xlabel('Verovatnoća (0.0 - 1.0)')
    plt.ylabel('Gustina podataka')
    plt.legend()
    plt.show()

    # ==============================
    # 5. STATISTIKA PREDIKCIJA
    # ==============================
    print("\n" + "="*30)
    print("FINALNI IZVEŠTAJ")
    print("="*30)
    print(f"RF Model - Accuracy: {rf_acc:.4f}")
    print(f"GBT Model - Accuracy: {gbt_acc:.4f}")
    print(f"Ukupno analiziranih studenata: {len(rf_df)}")
    print("="*30)

else:
    print("❌ Skripta nije mogla da pronađe ili pročita Parquet fajlove.")