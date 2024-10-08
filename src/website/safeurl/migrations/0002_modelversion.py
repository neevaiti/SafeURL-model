# Generated by Django 5.1.1 on 2024-10-04 02:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('safeurl', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelVersion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('version', models.CharField(max_length=255, unique=True)),
                ('model_path', models.TextField()),
                ('metrics', models.JSONField()),
                ('created_at', models.DateTimeField()),
            ],
        ),
    ]
