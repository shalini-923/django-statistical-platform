# Generated by Django 5.1.3 on 2024-11-19 07:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('csv_visualizer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='mlprediction',
            name='status',
            field=models.CharField(choices=[('success', 'Success'), ('failed', 'Failed')], default='success', max_length=50),
        ),
    ]
