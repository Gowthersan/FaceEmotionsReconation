# Emotion Detection Application

Cette application Django détecte les émotions à partir d'images en utilisant un modèle d'apprentissage profond.

## Fonctionnalités

- Détection d'émotions en temps réel via le flux caméra
- Classification en 7 émotions: colère, dégoût, peur, joie, tristesse, surprise, neutre
- Interface web intuitive

## Technologies

- **Backend**: Django 4.2
- **Machine Learning**: TensorFlow 2.16.1
- **Traitement d'image**: OpenCV 4.10.0
- **Autres**: NumPy, Pandas, scikit-learn

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/votre-utilisateur/emotion-detection.git
   cd emotion-detection
   ```

2. Créer et activer un environnement virtuel :

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirement.txt
   ```

## Exécution

Lancer l'application avec :

```powershell
.\run_app.ps1
```

L'application sera accessible à l'adresse : http://localhost:8000

## Structure du projet

```
emotion-detection/
├── emotion_app/          # Application principale
│   ├── ml_model/         # Modèle ML et entraînement
│   ├── templates/        # Templates HTML
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── utils.py
│   └── views.py
├── emotion_project/      # Configuration Django
├── static/               # Fichiers statiques (CSS, JS, images)
├── db.sqlite3            # Base de données
├── manage.py             # Utilitaire Django
└── requirement.txt       # Dépendances
```

## Contribution

Les contributions sont les bienvenues! Veuillez créer une issue pour discuter des changements proposés.
