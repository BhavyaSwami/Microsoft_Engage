from django.db import models
# Create your models here.
class Test(models.Model):
    username=models.CharField(max_length=20)
    email=models.CharField(max_length=100)
    password=models.CharField(max_length=150)
    private_key=models.CharField(max_length=30)