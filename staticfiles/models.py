from django.db import models

class Image(models.Model):
    file = models.ImageField(upload_to='uploads/')
    caption = models.CharField(max_length=255, blank=True, default='')

    def __str__(self):
        return self.file.name

