
# import pandas as pd
# import glob
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # 1. Putanje i podaci iz tvojih logova i Excel-a
# base_path = r'D:\02BIGDATA\results'
# final_results_path = os.path.join(base_path, 'final_results')

# models_info = {
#     'GBT Classifier': {'folder': 'gbt_final', 'accuracy': 70.79, 'train_time_cluster': 102.20, 'train_time_local': 80.04},
#     'Random Forest': {'folder': 'rf_final', 'accuracy': 68.56, 'train_time_cluster': 37.34, 'train_time_local': 21.62}
# }

# # --- DEO 1: MATRICE KONFUZIJE ---
# fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# fig.suptitle('Finalna Evaluacija Modela na Test Setu (2.8 miliona redova)', fontsize=16)

# for i, (model_name, info) in enumerate(models_info.items()):
#     search_path = os.path.join(final_results_path, info['folder'], "part-*")
#     all_files = glob.glob(search_path)
    
#     if not all_files:
#         print(f"Upozorenje: Nisu pronađeni fajlovi u {search_path}")
#         continue
    
#     print(f"Učitavam {len(all_files)} fajlova za {model_name}...")
#     df_list = [pd.read_parquet(f) for f in all_files]
#     df = pd.concat(df_list, ignore_index=True)
    
#     cm = confusion_matrix(df['label'], df['prediction'])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stress', 'Stress'])
#     disp.plot(cmap='Blues', ax=axes[i], values_format='d', colorbar=False)
#     axes[i].set_title(f'{model_name}\n(Accuracy: {info["accuracy"]}%)')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(os.path.join(base_path, '01_matrice_konfuzije.png'), dpi=300)

# # --- DEO 2: POREĐENJE TAČNOSTI (ACCURACY) ---
# plt.figure(figsize=(8, 6))
# model_names = list(models_info.keys())
# accuracies = [info['accuracy'] for info in models_info.values()]
# bars = plt.bar(model_names, accuracies, color=['steelblue', 'skyblue'])
# plt.ylim(60, 75)
# plt.ylabel('Tačnost (%)')
# plt.title('Uporedni prikaz tačnosti (Accuracy)')

# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

# plt.savefig(os.path.join(base_path, '02_accuracy_comparison.png'))

# # --- DEO 3: ANALIZA PERFORMANSI (LOKALNO VS KLASTER) ---
# # Podaci uzeti iz tvoje Excel tabele (Red 5 za RF i Red 9 za GBT)
# plt.figure(figsize=(10, 6))
# labels = ['GBT (D:6, I:25)', 'RF (T:50, D:6)']
# local_times = [info['train_time_local'] for info in models_info.values()]
# cluster_times = [info['train_time_cluster'] for info in models_info.values()]

# x = range(len(labels))
# width = 0.35

# plt.bar([p - width/2 for p in x], local_times, width, label='Lokalno (Single Node)', color='lightgreen')
# plt.bar([p + width/2 for p in x], cluster_times, width, label='Docker Klaster (Distributed)', color='salmon')

# plt.ylabel('Vreme treniranja (sekunde)')
# plt.title('Analiza performansi: Lokalno vs Docker Klaster')
# plt.xticks(x, labels)
# plt.legend()

# plt.savefig(os.path.join(base_path, '03_performance_analysis.png'))
# plt.show()

# print(f"\n=== SVE SLIKE SU SAČUVANE U: {base_path} ===")

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================
# 1. KONFIGURACIJA LOKALNIH PUTANJA
# ==============================================================
base_path = r'D:\02BIGDATA\results\final_results'

def read_local_spark_folder(folder_name, file_type='parquet'):
    """
    Spaja sve part fajlove iz Spark foldera u jedan Pandas DataFrame.
    """
    path = os.path.join(base_path, folder_name)
    
    if file_type == 'parquet':
        # Pandas može direktno da pročita ceo folder ako je validan Parquet dataset
        try:
            return pd.read_parquet(path)
        except:
            # Ako ne uspe, tražimo part fajlove ručno
            files = glob.glob(os.path.join(path, "part-*.parquet"))
            if not files: return None
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            
    elif file_type == 'csv':
        files = glob.glob(os.path.join(path, "part-*.csv"))
        if not files: return None
        # Spark CSV obično nema zaglavlja
        df_list = [pd.read_csv(f, header=None) for f in files]
        return pd.concat(df_list, ignore_index=True)

# ==============================================================
# 2. VREMENSKA ANALIZA (Dynamic)
# ==============================================================
def plot_temporal():
    print("Analiziram vremenske podatke...")
    df = read_local_spark_folder("analysis_monthly", file_type='csv')
    
    if df is not None:
        # Mapiramo kolone: 0:month, 1:accuracy, 2:samples
        df.columns = ['month', 'accuracy', 'samples']
        df = df.sort_values('month')
        df['month_str'] = df['month'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:]}")

        plt.figure(figsize=(12, 6))
        plt.plot(df['month_str'], df['accuracy'], marker='o', linewidth=2, color='#2ca02c')
        plt.fill_between(df['month_str'], df['accuracy'], alpha=0.2, color='#2ca02c')
        
        plt.title('Tačnost modela po mesecima (Dinamički iz lokalnih fajlova)', fontsize=14)
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, 'vremenska_analiza.png'))
        print("✅ Sačuvano: vremenska_analiza.png")

# ==============================================================
# 3. MATRICE KONFUZIJE (Dynamic)
# ==============================================================
def plot_matrices():
    print("Generišem matrice konfuzije...")
    models = {
        'Random Forest': 'rf_final',
        'GBT Classifier': 'gbt_final'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, (name, folder) in enumerate(models.items()):
        df = read_local_spark_folder(folder, file_type='parquet')
        
        if df is not None:
            # Računamo accuracy direktno iz podataka
            acc = (df['label'] == df['prediction']).mean() * 100
            cm = confusion_matrix(df['label'], df['prediction'])
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stress', 'Stress'])
            disp.plot(cmap='Blues', ax=axes[i], values_format='d', colorbar=False)
            axes[i].set_title(f'{name}\nReal-time Accuracy: {acc:.2f}%')
            print(f"✅ Obrađen {name}")

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'matrice_konfuzije_final.png'))
    print("✅ Sačuvano: matrice_konfuzije_final.png")

def plot_probability_distribution():
    print("Analiziram sigurnost predikcija (Probability)...")
    df = read_local_spark_folder("gbt_final", file_type='parquet')
    
    if df is not None:
        # Spark čuva probability kao vektor, izvlačimo maksimalnu verovatnoću
        # Pretpostavka: probability je lista/niz od 2 elementa [P(0), P(1)]
        df['max_prob'] = df['probability'].apply(lambda x: max(x))

        plt.figure(figsize=(10, 6))
        sns.histplot(df['max_prob'], bins=30, kde=True, color='purple')
        plt.title('Distribucija sigurnosti modela (Confidence)', fontsize=14)
        plt.xlabel('Verovatnoća (Sigurnost)')
        plt.ylabel('Broj uzoraka')
        plt.savefig(os.path.join(base_path, '05_confidence_dist.png'))
        print("✅ Sačuvano: 05_confidence_dist.png")

def plot_model_comparison():
    # Podaci iz tvog poslednjeg loga
    results = {
        'Model': ['Random Forest', 'GBT Classifier'],
        'Accuracy': [69.28, 72.19]
    }
    df_res = pd.DataFrame(results)

    plt.figure(figsize=(8, 6))
    bars = sns.barplot(x='Model', y='Accuracy', data=df_res, palette='viridis')
    plt.ylim(60, 80)
    plt.title('Finalni uporedni prikaz tačnosti', fontsize=14)
    
    for bar in bars.patches:
        plt.annotate(format(bar.get_height(), '.2f') + '%', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                     ha='center', va='center', size=12, xytext=(0, 8), 
                     textcoords='offset points', fontweight='bold')
                     
    plt.savefig(os.path.join(base_path, '06_model_comparison.png'))
    print("✅ Sačuvano: 06_model_comparison.png")
def plot_top_error_users():
    # Ručno unosimo top 5 sa tvog loga (ili učitaj iz foldera ako si sačuvao)
    bad_users = {
        'User ID (Skraćeno)': ['4228...', '7e22...', 'bd44...', 'ac02...', 'f5fa...'],
        'Accuracy': [14.88, 15.64, 16.38, 30.16, 30.25]
    }
    df_bad = pd.DataFrame(bad_users)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='User ID (Skraćeno)', data=df_bad, color='salmon')
    plt.axvline(x=72.19, color='red', linestyle='--', label='Prosečna tačnost')
    plt.title('Identifikacija Outlier-a: Korisnici kod kojih model najviše greši', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(base_path, '07_user_error_analysis.png'))
    print("✅ Sačuvano: 07_user_error_analysis.png")

def plot_feature_importance():
    print("Analiziram značajnost parametara (Feature Importance)...")
    
    df = read_local_spark_folder("importance_gbt", file_type='csv')
    
    if df is not None:
        # 1. Imenovanje kolona
        df.columns = ['Feature', 'Importance']
        
        # 2. POPRAVKA: Konverzija u numerički tip (veoma bitno!)
        df['Importance'] = pd.to_numeric(df['Importance'], errors='coerce')
        
        # 3. Sortiranje
        df = df.sort_values('Importance', ascending=True)

        plt.figure(figsize=(10, 8))
        
        # 4. Gradijent boja (sada će raditi jer su brojevi u pitanju)
        importance_norm = df['Importance'] / df['Importance'].max()
        colors = plt.cm.viridis(importance_norm)
        
        bars = plt.barh(df['Feature'], df['Importance'], color=colors)
        
        for i, v in enumerate(df['Importance']):
            plt.text(v + 0.005, i, f'{v*100:.1f}%', va='center', fontweight='bold', fontsize=10)

        plt.title('Značaj parametara u predikciji nivoa stresa (GBT Model)', fontsize=14)
        plt.xlabel('Koeficijent značajnosti (0.0 - 1.0)', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, '08_feature_importance_final.png'), dpi=300)
        print("✅ Sačuvano: 08_feature_importance_final.png")

if __name__ == "__main__":
    if not os.path.exists(base_path):
        print(f"GRESKA: Putanja {base_path} ne postoji!")
    else:
        plot_temporal()
        plot_matrices()
        plot_model_comparison()
        plot_probability_distribution()
        plot_top_error_users()
        # DODAJ OVO:
        plot_feature_importance() 
        
        print("\nSve vizuelizacije su završene!")
        plt.show()
# # ==============================================================
# # IZVRŠAVANJE
# # ==============================================================
# if __name__ == "__main__":
#     # Provera da li putanja postoji
#     if not os.path.exists(base_path):
#         print(f"GRESKA: Putanja {base_path} ne postoji!")
#     else:
#         plot_temporal()
#         plot_matrices()
#         plot_model_comparison()
#         plot_probability_distribution()
#         plot_top_error_users()
#         print("\nSve vizuelizacije su završene!")
#         plt.show()

# ==============================================================
# IZVRŠAVANJE
# ==============================================================
# if __name__ == "__main__":
#     # Provera da li putanja postoji
#     if not os.path.exists(base_path):
#         print(f"GRESKA: Putanja {base_path} ne postoji!")
#     else:
#         print("\n" + "="*60)
#         print("UČITAVANJE REZULTATA PREDIKCIJE ZA OBA MODELA")
#         print("="*60)

#         # 1. PRIKAZ ZA RANDOM FOREST
#         print("\n>>> RANDOM FOREST (rf_final):")
#         df_rf = read_local_spark_folder("rf_final", file_type='parquet')
#         # Izmeni ovaj deo u visualization_final_inference.py
#         if df_rf is not None:
#              # Prikazujemo samo ono što postoji u fajlu
#              print(df_rf[['label', 'prediction']].head(10))
#         else:
#             print("Greška: Ne mogu da učitam rf_final.")

#         print("\n" + "-"*40)

#         # 2. PRIKAZ ZA GBT CLASSIFIER
#         print(">>> GBT CLASSIFIER (gbt_final):")
#         df_gbt = read_local_spark_folder("gbt_final", file_type='parquet')
#         if df_gbt is not None:
#             # Prikazujemo UID, Day, stvarni Label i šta je GBT predvideo
#             print(df_gbt[['uid', 'day', 'label', 'prediction']].head(10))
#         else:
#             print("Greška: Ne mogu da učitam gbt_final.")

#         print("="*60 + "\n")

#         # Pokretanje vizuelizacija (grafika)
#         plot_temporal()
#         plot_matrices()
        
#         print("\n✅ Sve operacije su uspešno završene!")
#         plt.show()