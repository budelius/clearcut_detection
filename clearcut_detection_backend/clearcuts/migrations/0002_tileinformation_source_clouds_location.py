# Generated by Django 2.2.3 on 2020-05-14 07:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('clearcuts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='tileinformation',
            name='source_clouds_location',
            field=models.CharField(blank=True, max_length=60, null=True),
        ),
    ]