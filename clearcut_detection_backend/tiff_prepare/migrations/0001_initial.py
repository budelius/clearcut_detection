# Generated by Django 2.2.14 on 2020-08-22 07:39

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('clearcuts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Prepared',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_date', models.DateField()),
                ('model_tiff_location', models.URLField(blank=True, null=True)),
                ('cloud_tiff_location', models.URLField(blank=True, null=True)),
                ('success', models.SmallIntegerField(default=0)),
                ('prepare_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('tile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tiff_prepare', to='clearcuts.Tile')),
            ],
            options={
                'unique_together': {('tile', 'image_date')},
            },
        ),
    ]
