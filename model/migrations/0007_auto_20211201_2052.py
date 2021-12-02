# Generated by Django 3.2.5 on 2021-12-02 01:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0006_feedbck'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Feedbck',
            new_name='Feedback',
        ),
        migrations.AlterField(
            model_name='feedback',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='model.doctoruser'),
        ),
    ]
