from django import forms

class HeartPredictionForm(forms.Form):

    SEX_CHOICES = [(1, 'Homme'), (0, 'Femme')]
    CP_CHOICES = [(1, 'Angine typique'), (2, 'Angine atypique'), (3, 'Douleur non-angineuse'), (4, 'Asymptomatique')]
    FBS_CHOICES = [(1, '> 120 mg/dl'), (0, '< 120 mg/dl')]
    RESTECG_CHOICES = [(0, 'Normal'), (1, 'Anomalie onde ST-T'), (2, 'Hypertrophie ventriculaire')]
    EXANG_CHOICES = [(1, 'Oui'), (0, 'Non')]
    SLOPE_CHOICES = [(1, 'Montante'), (2, 'Plate'), (3, 'Descendante')]
    THAL_CHOICES = [(3, 'Normal'), (6, 'Défaut fixé'), (7, 'Défaut réversible')]

    age = forms.IntegerField(label="Âge du patient", min_value=1, max_value=120)
    sex = forms.ChoiceField(label="Sexe", choices=SEX_CHOICES)
    cp = forms.ChoiceField(label="Type de douleur thoracique", choices=CP_CHOICES)
    trestbps = forms.IntegerField(label="Pression artérielle au repos (mm Hg)")
    chol = forms.IntegerField(label="Cholestérol sérique (mg/dl)")
    fbs = forms.ChoiceField(label="Glycémie à jeun > 120 mg/dl", choices=FBS_CHOICES)
    restecg = forms.ChoiceField(label="Résultat ECG au repos", choices=RESTECG_CHOICES)
    thalach = forms.IntegerField(label="Fréquence cardiaque maximale atteinte")
    exang = forms.ChoiceField(label="Angine déclenchée par l'effort", choices=EXANG_CHOICES)
    oldpeak = forms.FloatField(label="Dépression ST induite par l'effort")
    slope = forms.ChoiceField(label="Pente du segment ST à l'effort", choices=SLOPE_CHOICES)
    ca = forms.IntegerField(label="Nombre de vaisseaux majeurs colorés (0-3)", min_value=0, max_value=3)
    thal = forms.ChoiceField(label="Résultat du test de thalassémie", choices=THAL_CHOICES)