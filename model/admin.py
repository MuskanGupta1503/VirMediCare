from django.contrib import admin
from model.models import *
# Register your models here.

admin.site.register(PatientUser)
admin.site.register(DoctorUser)
admin.site.register(Appointment)

