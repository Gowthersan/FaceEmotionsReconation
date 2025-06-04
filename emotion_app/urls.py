from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.process_frame, name='detect_emotion'),
]
