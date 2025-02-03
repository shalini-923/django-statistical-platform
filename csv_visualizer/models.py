import hashlib
from django.db import models
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
import pandas as pd


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    ROLE_CHOICES = [
        ('analyst', 'Analyst'),
        ('viewer', 'Viewer'),
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='viewer')

    def __str__(self):
        return f"{self.user.username} - {self.role}"
    
# File size limit in bytes (10 MB)
MAX_FILE_SIZE = 10485760  # 10 MB



def get_file_hash(file):
    """Generate a hash for the file's contents."""
    hash_sha256 = hashlib.sha256()
    for chunk in file.chunks():  # Read file in chunks
        hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def validate_csv_file(file):
    """Ensure the file is a CSV file."""
    if not file.name.endswith('.csv'):
        raise ValidationError("Only CSV files are allowed.")
    
     # Check if file size exceeds the limit
    if file.size > MAX_FILE_SIZE:
        raise ValidationError(f"File size cannot exceed {MAX_FILE_SIZE / 1048576} MB.")

class CSVFile(models.Model):
   
    file = models.FileField(upload_to='csvs/', validators=[validate_csv_file])
    file_hash = models.CharField(max_length=64, unique=True, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE) 


    def __str__(self):
        return self.file.name

    def save(self, *args, **kwargs):
        # Automatically calculate the hash if the file is new or changed
        if not self.file_hash:
            self.file_hash = get_file_hash(self.file)
        
        # Check if a file with the same hash already exists
        if CSVFile.objects.filter(file_hash=self.file_hash).exists():
            raise ValidationError("This file has already been uploaded.")
        
        super().save(*args, **kwargs)

        
    def perform_analysis(self):
        try:
            # Perform analysis using pandas
            file_path = self.file.path
            data = pd.read_csv(file_path)

            # Check if the file contains a valid header row
            if data.empty or not all(isinstance(col, str) for col in data.columns):
                raise ValidationError("The CSV file must have a valid header row.")

            # Check for missing values
            if data.isnull().sum().sum() > 0:
                raise ValidationError("The CSV file contains missing values.")

            # Validate numeric fields (e.g., ensure they are within a reasonable range)
            for column in data.select_dtypes(include=['float64', 'int64']).columns:
                if data[column].min() < 0 or data[column].max() > 1000000:  # Example range check
                    raise ValidationError(f"Invalid values in column '{column}' within the allowed range.")


            # Example analysis: summary stats
            summary_stats = data.describe().to_dict()

            # Create an analysis result entry in the database
            AnalysisResult.objects.create(
                data_upload=self,
                summary_stats=summary_stats
            )
        except Exception as e:
            # Handle any exceptions that occur during analysis (e.g., invalid CSV format)
            raise ValidationError(f"Error while performing analysis: {str(e)}")

class AnalysisResult(models.Model):
    data_upload = models.ForeignKey(CSVFile, on_delete=models.CASCADE)
    summary_stats = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.data_upload.file.name}"

class MLPrediction(models.Model):
    data_upload = models.ForeignKey(CSVFile, on_delete=models.CASCADE)
    predictions = models.JSONField()
    model_name = models.CharField(max_length=100)
    status = models.CharField(max_length=50, choices=[('success', 'Success'), ('failed', 'Failed')], default='success')
    created_at = models.DateTimeField(auto_now_add=True)
    visualization = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Prediction by {self.model_name} for {self.data_upload.file.name}"
