# Generated by Django 3.2.5 on 2021-12-01 04:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0003_auto_20211130_1909'),
    ]

    operations = [
        migrations.CreateModel(
            name='Appointment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('applydate', models.DateField()),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='model.doctoruser')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='model.patientuser')),
            ],
        ),
    ]
