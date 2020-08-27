# Generated by Django 2.2.14 on 2020-08-21 21:43

import django.contrib.gis.db.models.fields
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Tile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tile_index', models.CharField(max_length=7, unique=True)),
                ('is_tracked', models.SmallIntegerField(default=0)),
                ('first_date', models.DateField(default=None, null=True)),
                ('last_date', models.DateField(default=None, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Zone',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tile', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='tile_zones', to='clearcuts.Tile')),
            ],
        ),
        migrations.CreateModel(
            name='TileInformation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tile_name', models.CharField(max_length=7)),
                ('tile_date', models.DateField(default=django.utils.timezone.now)),
                ('tile_location', models.URLField(blank=True, null=True)),
                ('source_tci_location', models.URLField(blank=True, null=True)),
                ('source_b04_location', models.URLField(blank=True, null=True)),
                ('source_b08_location', models.URLField(blank=True, null=True)),
                ('source_b8a_location', models.URLField(blank=True, null=True)),
                ('source_b11_location', models.URLField(blank=True, null=True)),
                ('source_b12_location', models.URLField(blank=True, null=True)),
                ('source_clouds_location', models.URLField(blank=True, null=True)),
                ('model_tiff_location', models.URLField(blank=True, null=True)),
                ('tile_metadata_hash', models.CharField(blank=True, default=0, max_length=32, null=True)),
                ('cloud_coverage', models.FloatField(default=0)),
                ('mapbox_tile_id', models.CharField(blank=True, max_length=32, null=True)),
                ('mapbox_tile_name', models.CharField(blank=True, max_length=32, null=True)),
                ('mapbox_tile_layer', models.CharField(blank=True, max_length=32, null=True)),
                ('coordinates', django.contrib.gis.db.models.fields.PolygonField(blank=True, null=True, srid=4326)),
                ('is_downloaded', models.SmallIntegerField(default=0)),
                ('is_prepared', models.SmallIntegerField(default=0)),
                ('is_predicted', models.SmallIntegerField(default=0)),
                ('is_converted', models.SmallIntegerField(default=0)),
                ('is_uploaded', models.SmallIntegerField(default=0)),
                ('tile_index', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tile_information', to='clearcuts.Tile')),
            ],
        ),
        migrations.CreateModel(
            name='RunUpdateTask',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path_type', models.SmallIntegerField(default=0)),
                ('path_img_0', models.URLField(default='')),
                ('path_img_1', models.URLField(default='')),
                ('image_date_0', models.DateField()),
                ('image_date_1', models.DateField()),
                ('path_clouds_0', models.URLField(default='')),
                ('path_clouds_1', models.URLField(default='')),
                ('result', models.URLField(null=True)),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('date_started', models.DateTimeField(default=None, null=True)),
                ('date_finished', models.DateTimeField(default=None, null=True)),
                ('tile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='run_update_task', to='clearcuts.Tile')),
            ],
            options={
                'db_table': 'clearcuts_run_update_task',
            },
        ),
        migrations.CreateModel(
            name='NotClearcut',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_date_previous', models.DateField(null=True)),
                ('image_date_current', models.DateField(default=django.utils.timezone.now)),
                ('area', models.FloatField()),
                ('forest', models.PositiveIntegerField(default=1)),
                ('clouds', models.PositiveIntegerField(default=0)),
                ('centroid', django.contrib.gis.db.models.fields.PointField(srid=4326)),
                ('mpoly', django.contrib.gis.db.models.fields.PolygonField(srid=4326)),
                ('zone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='clearcuts.Zone')),
            ],
        ),
        migrations.CreateModel(
            name='Clearcut',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_date_previous', models.DateField(null=True)),
                ('image_date_current', models.DateField(default=django.utils.timezone.now)),
                ('area', models.FloatField()),
                ('forest', models.PositiveIntegerField(default=1)),
                ('clouds', models.PositiveIntegerField(default=0)),
                ('centroid', django.contrib.gis.db.models.fields.PointField(srid=4326)),
                ('mpoly', django.contrib.gis.db.models.fields.PolygonField(geography=True, srid=4326)),
                ('zone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='clearcuts.Zone')),
            ],
        ),
    ]
