import os

import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model

# Construction du chemin absolu vers le modèle
model_path = os.path.join(settings.BASE_DIR, 'emotion_app', 'ml_model', 'fer_model.h5')

# Vérification que le fichier modèle existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle est introuvable à l'emplacement: {model_path}")

# Chargement du modèle
model = load_model(model_path)
emotion_labels = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Tristesse', 'Surprise', 'Neutre']

# Détection des émotions
def detect_emotion(frame):
    # Préprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    
    # Prédiction
    predictions = model.predict(reshaped)[0]
    emotion_idx = np.argmax(predictions)
    return emotion_labels[emotion_idx], float(predictions[emotion_idx])
