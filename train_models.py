import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
import joblib

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("F1-Score :", f1_score(y_test, y_pred))
    return model

def main():
    # Chargement des données
    try:
        X_train = pd.read_csv("X_train_scaled.csv")
        X_test = pd.read_csv("X_test_scaled.csv")
        y_train = pd.read_csv("y_train.csv").squeeze()
        y_test = pd.read_csv("y_test.csv").squeeze()
    except FileNotFoundError:
        print("Les fichiers de données (X_train_scaled.csv, etc.) sont introuvables.")
        print("Veuillez d'abord exécuter main.py pour les générer.")
        return

    # Définition des modèles
    models = {
        "Arbre de Décision": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Noms de fichiers pour la sauvegarde
    save_names = {
        "Arbre de Décision": "modele_coeur_arbre_decision.pkl",
        "KNN": "modele_coeur_knn.pkl",
        "Régression Logistique": "modele_coeur_logistique.pkl"
    }

    # Entraînement, évaluation et sauvegarde
    for name, model_cls in models.items():
        trained_model = evaluate_model(name, model_cls, X_train, y_train, X_test, y_test)
        joblib.dump(trained_model, save_names[name])
        print(f"Modèle sauvegardé sous : {save_names[name]}")

if __name__ == "__main__":
    main()
