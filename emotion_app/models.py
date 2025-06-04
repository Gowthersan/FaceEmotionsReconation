from django.db import models


class EmotionRecord(models.Model):
    image = models.ImageField(upload_to='emotion_images/')
    emotion = models.CharField(max_length=20)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.emotion} ({self.confidence:.2f}) - {self.created_at}"
