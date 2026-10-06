import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================================
# 1) CHARGEMENT DU DATASET
# ======================================================

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv("processed.cleveland.data", names=columns, na_values="?")

print("Dimensions initiales :", df.shape)

# ======================================================
# 2) VALEURS MANQUANTES
# ======================================================

print("\nValeurs manquantes AVANT :")
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))

print("\nValeurs manquantes APRÈS :")
print(df.isnull().sum())

# ======================================================
# 3) DOUBLONS
# ======================================================

duplicates_before = df.duplicated().sum()
print(f"\nDoublons AVANT : {duplicates_before}")

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()
print(f"Doublons APRÈS : {duplicates_after}")

# ======================================================
# 4) TRANSFORMATION DE LA TARGET
# ======================================================

df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# ======================================================
# 5) SAUVEGARDE DU DATASET CLEAN FINAL
# ======================================================

df.to_csv("heart_disease_clean.csv", index=False)
print("\nDataset clean final sauvegardé : heart_disease_clean.csv")

# ======================================================
# 6) SÉPARATION X / y
# ======================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nX shape :", X.shape)
print("y shape :", y.shape)

# ======================================================
# 7) TRAIN / TEST
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain :", X_train.shape)
print("Test :", X_test.shape)

# ======================================================
# 8) STANDARDISATION
# ======================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardisation terminée !")

# ======================================================
# 9) SAUVEGARDE DES DONNÉES
# ======================================================

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("X_test_scaled.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

joblib.dump(scaler, "scaler.pkl")

print("\nFichiers sauvegardés :")
print("- heart_disease_clean.csv")
print("- X_train_scaled.csv")
print("- X_test_scaled.csv")
print("- y_train.csv")
print("- y_test.csv")
print("- scaler.pkl")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CHARGEMENT DU DATASET CLEAN

df = pd.read_csv("heart_disease_clean.csv")

print("Dimensions du dataset :", df.shape)
print("-" * 50)

# 1) STATISTIQUES DESCRIPTIVES

print("\n--- 1) STATISTIQUES DESCRIPTIVES ---")

stats = df.describe().T
print(stats)

# 2) CORRÉLATIONS

print("\nGÉNÉRATION DE LA MATRICE DE CORRÉLATION")
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de Corrélation entre les variables", fontsize=14)
plt.tight_layout()
plt.savefig("eda_1_correlation_matrix.png")
# plt.show()

#3) DISTRIBUTION DE L'ÂGE SELON LA TARGET

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, 
    x='age', 
    hue='target', 
    multiple='dodge', 
    palette=['#3498db', '#e74c3c'], 
    bins=15, 
    edgecolor='black'
)
plt.title("Répartition de la maladie cardiaque selon l'âge", fontsize=14, pad=15)
plt.xlabel("Âge des patients", fontsize=12)
plt.ylabel("Nombre de patients", fontsize=12)

# Personnalisation de la légende
plt.legend(title="Diagnostic", labels=['Malade (1)', 'Sain (0)'])

plt.tight_layout()
plt.savefig("eda_3_age_vs_target_claire.png")
# plt.show()

# 4) DÉSÉQUILIBRE DES CLASSES (Target)
print("\n 4) ANALYSE DU DÉSÉQUILIBRE DES CLASSES")
# Comptage en valeur absolue
class_counts = df["target"].value_counts()
print("Nombre de patients par classe :")
print(class_counts)

# Comptage en pourcentage
class_percentages = df["target"].value_counts(normalize=True) * 100
print("\nPourcentage par classe (%) :")
print(class_percentages)

# Visualisation du déséquilibre
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df, palette="Set2")
plt.title("Répartition de la variable cible (0 = Sain, 1 = Malade)")
plt.xlabel("Target")
plt.ylabel("Nombre de patients")
plt.tight_layout()
plt.savefig("eda_4_class_imbalance.png")
# plt.show()

print("\n EDA TERMINÉE")