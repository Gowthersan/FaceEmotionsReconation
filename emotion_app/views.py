import base64

import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .utils import detect_emotion


def index(request):
    return render(request, 'detector/index.html')

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        # Récupération de l'image depuis le frontend
        img_data = request.POST.get('image').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Détection d'émotion
        emotion, confidence = detect_emotion(frame)
        
        return JsonResponse({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2)
        })
    return JsonResponse({'error': 'Méthode non autorisée'}, status=400)
