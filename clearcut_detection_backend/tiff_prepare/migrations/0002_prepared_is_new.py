# Generated by Django 3.1 on 2020-08-31 17:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tiff_prepare', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='prepared',
            name='is_new',
            field=models.SmallIntegerField(default=1),
        ),
    ]
