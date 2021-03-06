"""manage_health URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from model.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index,name="index"),
    path('predictDisease',predictDisease,name="predictDisease"),
    path('user_login',user_login,name="user_login"),
    path('user_signup',user_signup,name="user_signup"),
    path('user_home',user_home,name="user_home"),
    path('doctor_login',doctor_login,name="doctor_login"),
    path('doctor_signup',doctor_signup,name="doctor_signup"),
    path('doctor_home',doctor_home,name="doctor_home"),
    path('Logout',Logout,name="Logout"),
    path('doctor_search',doctor_search,name="doctor_search"),
    path('book_appointment/<int:pid>',book_appointment,name="book_appointment"),
    path('doctor_details/<int:pid>',doctor_details,name="doctor_details"),
    path('self_examine',self_examine,name="self_examine"),
    path('heartdis',heartdis,name="heartdis"),
    path('thyroism',thyroism,name="thyroism"),
    path('diabetesr',diabetesr,name="diabetesr"),
    path('healthnews',healthnews,name="healthnews"),
    path('feedback',feedback,name="feedback"),
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
