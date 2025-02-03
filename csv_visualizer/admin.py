# data_app/admin.py
from django.contrib import admin
from .models import UserProfile, CSVFile, AnalysisResult, MLPrediction

admin.site.register(UserProfile)
admin.site.register(CSVFile)
admin.site.register(AnalysisResult)
admin.site.register(MLPrediction)
