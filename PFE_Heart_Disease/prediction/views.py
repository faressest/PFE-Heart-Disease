from django.shortcuts import render
from .forms import HeartPredictionForm
import joblib
import numpy as np
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, 'ml_models', 'modele_coeur_knn.pkl')
SCALER_PATH = os.path.join(MODULE_DIR, 'ml_models', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Erreur lors du chargement des fichiers .pkl : {e}")

def home(request):
    return render(request, 'prediction/home.html')

def predict_heart(request):
    result = None
    probability = 0  
    input_summary = None
    form = HeartPredictionForm()
    if request.method == 'POST':
        form = HeartPredictionForm(request.POST)
        if form.is_valid():
            try:
                input_data = [
                    form.cleaned_data['age'],
                    int(form.cleaned_data['sex']),
                    int(form.cleaned_data['cp']),
                    form.cleaned_data['trestbps'],
                    form.cleaned_data['chol'],
                    int(form.cleaned_data['fbs']),
                    int(form.cleaned_data['restecg']),
                    form.cleaned_data['thalach'],
                    int(form.cleaned_data['exang']),
                    form.cleaned_data['oldpeak'],
                    int(form.cleaned_data['slope']),
                    form.cleaned_data['ca'],
                    int(form.cleaned_data['thal'])
                ]
                import pandas as pd
                columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                features_df = pd.DataFrame([input_data], columns=columns)
                
                scaled_features = scaler.transform(features_df)
                prediction = int(model.predict(scaled_features)[0])
                try:
                    probs = model.predict_proba(scaled_features)[0]
                    probability = int(round(probs[1] * 100))
                    if prediction == 1 and probability < 50:
                        probability = 75 
                    elif prediction == 0 and probability > 50:
                        probability = 25 
                except:
                    probability = 100 if prediction == 1 else 0
                if prediction == 1:
                    result = "Attention : Présence probable d'une maladie cardiaque, veuillez consulter notre clinique IDIA au plus tot possible."
                else:
                    result = "Félicitations : Aucun signe de maladie cardiaque détecté."
                
                input_summary = form.cleaned_data
            except Exception as e:
                result = f"Erreur lors du traitement : {e}"
    return render(request, 'prediction/index.html', {
        'form': form, 
        'result': result, 
        'probability': probability,
        'input_summary': input_summary
    })

