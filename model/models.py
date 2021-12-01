from django.db import models
from django.contrib.auth.models import User
from django.db.models.deletion import CASCADE
# Create your models here.

class PatientUser(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    mobile = models.CharField(max_length=15,null=True)
    dob=models.DateField()
    image = models.FileField(null=True)
    gender = models.CharField(max_length=10,null=True)
    type = models.CharField(max_length=15,null=True)
    bloodgroup=models.CharField(max_length=3,null=True)
    location=models.CharField(max_length=50,null=True)
    def _str_(self):
        return self.user.username

class DoctorUser(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    mobile = models.CharField(max_length=15,null=True)
    dob=models.DateField()
    image = models.FileField(null=True)
    gender = models.CharField(max_length=10,null=True)
    type = models.CharField(max_length=15,null=True)
    status = models.CharField(max_length=20,null=True)
    location=models.CharField(max_length=50,null=True)
    specialization=models.CharField(max_length=30,null=True)
    remarks=models.CharField(max_length=60,null=True)
    def _str_(self):
        return self.user.username

class Appointment(models.Model):
    patient = models.ForeignKey(PatientUser,on_delete=models.CASCADE)
    doctor = models.ForeignKey(DoctorUser,on_delete=models.CASCADE)
    appointmentdate = models.DateField()
    def _str_(self):
        return self.id 