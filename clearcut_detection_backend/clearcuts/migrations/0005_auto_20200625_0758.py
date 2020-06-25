# Generated by Django 2.2.3 on 2020-06-25 07:58

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('clearcuts', '0004_tileinformation_tile_date'),
    ]

    operations = [
        migrations.AlterField(
            model_name='clearcut',
            name='image_date_0',
            field=models.DateField(default=datetime.datetime(2020, 6, 25, 7, 57, 54, 386399, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='clearcut',
            name='image_date_1',
            field=models.DateField(default=datetime.datetime(2020, 6, 25, 7, 57, 54, 386504, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='tile_date',
            field=models.DateField(default=datetime.datetime(2020, 6, 25, 7, 57, 54, 388930, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='tile_location',
            field=models.URLField(blank=True, null=True),
        ),
    ]
