import re
from django.shortcuts import render,redirect
from django.core.checks.messages import Error
from django.contrib.auth.models import User
from .models import *
from django.contrib.auth import authenticate,login, logout
from datetime import date 
from django.core.paginator import Paginator

# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from Disease_Diagnosis.utils import  heart_disease, diabetes,thyroid

# Create your views here.
def predict(syptomToBeTested):
    # DATA_PATH = "/Training.csv"
    data = pd.read_csv(r'C:\Users\Hello\Desktop\pythonfiles\MyProject-Health-Manage\dataset\Training.csv').dropna(axis = 1)
    
    # Encoding the target value into numerical
    # value using LabelEncoder
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])
    
    #Spliting data
    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size = 0.2, random_state = 24)

    #models
    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)

    # Reading the test data
    test_data = pd.read_csv(r'C:\Users\Hello\Desktop\pythonfiles\MyProject-Health-Manage\dataset\Testing.csv').dropna(axis=1)
    test_X = test_data.iloc[:, :-1]
    test_Y = encoder.transform(test_data.iloc[:, -1])

    symptoms = X.columns.values
 
    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index
    
    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }

    def predictDisease(symptoms):
        symptoms = symptoms.split(",")
        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        # reshaping the input data and converting it
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
        # generating individual outputs
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
        # making final prediction by taking mode of all predictions
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": nb_prediction,
            "final_prediction":final_prediction}
        return predictions["final_prediction"]
    return predictDisease(syptomToBeTested)

def index(request):
    return render(request,"index.html")

def predictDisease(request):
    if request.method=='POST':
        s1=request.POST['Symptom1']
        s2=request.POST['Symptom2']
        s3=request.POST['Symptom3']
        s4=request.POST['Symptom4']
        s5=request.POST['Symptom5']
        print(s1,s2,s3,s4,s5)
        syptomToBeTested=s1+","+s2+","+s3+","+s4+","+s5
        print(syptomToBeTested)
        result=predict(syptomToBeTested)
        d={'result':result}
        return render(request,'predictDisease.html',d)
    return render(request,'predictDisease.html')

def user_login(request):
    error=""
    if request.method=='POST':
        u=request.POST['uname']
        p=request.POST['pwd']
        user=authenticate(username=u,password=p)
        if user:
            try:
                user1=PatientUser.objects.get(user=user)
                if user1.type=="patient":
                    login(request,user)
                    error="no"
                else:
                    error="yes"
            except:
                error="yes"
        else:
            error="yes"
    d={'error':error}
    return render(request,'user_login.html',d)

def user_signup(request):
    error=""
    if request.method=='POST':
        f=request.POST['fname']
        l=request.POST['lname']
        i=request.FILES['image']
        p=request.POST['pwd']
        e=request.POST['email']
        dob=request.POST['dob']
        con=request.POST['contact']
        gen=request.POST['gender']
        print(f,l,e)
        try:
            print("Enter Try")
            user=User.objects.create_user(first_name=f, last_name=l, username=e, password=p)
            print("user created")
            PatientUser.objects.create(user=user, mobile=con,dob=dob,image=i,gender=gen,type="patient",bloodgroup="A+")
            print("Patient created")
            error="no"
        except:
            error="yes"
    d={'error':error}
    return render(request,'user_signup.html',d)

def user_home(request):
    if not request.user.is_authenticated:
        return redirect('user_login')
    return render(request,'user_home.html')

def doctor_login(request):
    error=""
    if request.method=='POST':
        u=request.POST['uname']
        p=request.POST['pwd']
        user=authenticate(username=u,password=p)
        if user:
            try:
                user1=DoctorUser.objects.get(user=user)
                if user1.type=="doctor" and user1.status!="pending":
                    login(request,user)
                    error="no"
                else:
                    error="not"
            except:
                error="yes"
        else:
            error="yes"
    d={'error':error}
    return render(request,'doctor_login.html',d)

def doctor_signup(request):
    error=""
    if request.method=='POST':
        f=request.POST['fname']
        l=request.POST['lname']
        i=request.FILES['image']
        p=request.POST['pwd']
        e=request.POST['email']
        con=request.POST['contact']
        gen=request.POST['gender']
        location=request.POST['location']
        dob=request.POST['dob']
        specialization=request.POST['specialization']
        remarks=request.POST['remarks']
        try:
            print("enter try")
            user=User.objects.create_user(first_name=f, last_name=l, username=e, password=p)
            print("user model created")
            DoctorUser.objects.create(user=user, mobile=con,dob=dob,image=i,
            gender=gen, type="doctor", status="pending",location=location,
            specialization=specialization,remarks=remarks)
            print("doctor created")
            error="no"
        except:
            error="yes"
    d={'error':error}
    return render(request,'doctor_signup.html',d)

def doctor_home(request):
    if not request.user.is_authenticated:
        return redirect('doctor_login')
    return render(request,'doctor_home.html')

def Logout(request):
    logout(request)
    return redirect('index')

def doctor_search(request):
    if not request.user.is_authenticated:
        return redirect('user_login')
    doctor = DoctorUser.objects.all()
    paginator = Paginator(doctor, 3) # Show 25 contacts per page.
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'doctor_search.html', {'page_obj': page_obj})

def doctor_details(request,pid):
    doctor = DoctorUser.objects.all().get(id=pid)
    d = {'doctor':doctor}
    return render(request,'doctor_details.html',d)

def book_appointment(request,pid):
    if not request.user.is_authenticated:
        return redirect('user_login')
    error=""
    user=request.user
    patient=PatientUser.objects.get(user=user)
    doctor=DoctorUser.objects.get(id=pid)
    if request.method=='POST':
        try:
            appointmentdate=request.POST['appointmentdate']
            Appointment.objects.create(patient=patient,doctor=doctor,appointmentdate=appointmentdate)  
            error="no"     
        except:
            error="yes"
    d={'error':error,'doctor':doctor}
    return render(request,'book_appointment.html',d)

'''
Input Features-
Pregnancies - number of times person has been pregnant
Glucose - glucose levels in Hg 
BloodPressure - blood pressure in mm hg
SkinThickness - skin thickness in mm
Insulin - insulin levels in mIU/L
BMI - Body Mass Index in Kg/m2
DiabetesPedigreeFunction - likelihood of diabetes based on family history
Age - age in years
sample input = [1, 89, 66, 23, 94, 28.1, 0.167, 21]
'''
def diabetesr(request):
    if request.method=='POST':
        pragnancies = request.POST.get('prag')
        glucose = request.POST.get('glucose')
        bp = request.POST.get('bp')
        skinthickness = request.POST.get('skinthick')
        insulin = request.POST.get('insulin')
        bmi = request.POST.get('bmi')
        dpf = request.POST.get('DPF')
        age = request.POST.get('age')
        res = diabetes([pragnancies,glucose,bp,skinthickness,insulin,bmi,dpf,age])
        context={
            'data':res
        }
        return render(request,'diabetes.html',context=context)
    return render(request,'diabetes.html')

'''
Input Features-
Age - age in years
Sex - (1 = male, 0 = female)
On_thyroxine - (1 = yes, 0 = no)
Query_on_thyroxine - (1 = yes, 0 = no)
On_antithyroid_medication - (1 = yes, 0 = no)
Sick - (1 = yes, 0 = no)
Pregnant - (1 = yes, 0 = no)
Thyroid_surgery - (1 = yes, 0 = no)
I131_treatment - (1 = yes, 0 = no)
Query_hypothyroid - (1 = yes, 0 = no)
Query_hyperthyroid - (1 = yes, 0 = no)
Lithium - (1 = yes, 0 = no)
Goiter - (1 = yes, 0 = no)
Tumor - (1 = yes, 0 = no)
Hypopituitary - (1 = yes, 0 = no)
Psych - (1 = yes, 0 = no)
TSH - TSH levels in mIU/mL
T3 - T3 levels in pg/dl
TT4 - TT4 levels in ng/dl
T4U - Thyroxine Utilization Rates
FTI - Free Thyroxine Index
sample input = [24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00025,0.03,0.143,0.133,0.108]
'''
def thyroism(request):
    if request.method=='POST':
        Age = int(request.POST.get('age'))
        Sex = int(request.POST.get('sex'))
        On_thyroxine = int(request.POST.get('On_thyroxine'))
        Query_on_thyroxine = int(request.POST.get('Query_on_thyroxine'))
        On_antithyroid_medication = int(request.POST.get('On_antithyroid_medication'))
        Sick = int(request.POST.get('Sick'))
        Pregnant = int(request.POST.get('Pregnant'))
        Thyroid_surgery = int(request.POST.get('Thyroid_surgery'))
        I131_treatment = int(request.POST.get('I131_treatment'))
        Query_hypothyroid = int(request.POST.get('Query_hypothyroid'))
        Query_hyperthyroid = int(request.POST.get('Query_hyperthyroid'))
        Lithium = int(request.POST.get('Lithium'))
        Goiter = int(request.POST.get('Goiter'))
        Tumor = int(request.POST.get('Tumor'))
        Hypopituitary = int(request.POST.get('Hypopituitary'))
        Psych = int(request.POST.get('Psych'))
        TSHh = float(request.POST.get('TSH'))
        T3 = float(request.POST.get('T3'))
        TT4h = float(request.POST.get('TT4'))
        T4Uh = float(request.POST.get('T4U'))
        FTIh = float(request.POST.get('FTI'))
        res = thyroid([Age,Sex,On_thyroxine,Query_on_thyroxine,On_antithyroid_medication,Sick,Pregnant,Thyroid_surgery,I131_treatment,Query_hypothyroid,Query_hyperthyroid,Lithium,Goiter,Tumor,Hypopituitary,Psych,TSHh,T3,TT4h,T4Uh,FTIh])
        context ={
            'data':res
        }
        return render(request,'thyroidismDetection.html',context)
    return render(request,'thyroidismDetection.html')

'''
Input Features-
age - age in years
sex - (1 = male, 0 = female)
cp - chest pain type (0 = No pain, 1 = low, 2 = moderate, 3 = Severe)
trestbps - resting blood pressure (in mm Hg)
chol - serum cholestoral in mg/dl
fbs - fasting blood sugar (1 = greator that 120 mg/dl; 0 = lesser that 120 mg/dl)
restecg - resting electrocardiographic results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes, 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)
ca - number of major vessels (0-3) colored by flourosopy
thal - (1 = normal; 2 = fixed defect; 3 = reversable defect)
sample input = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
'''
def heartdis(request):
    if request.method=='POST':
        Age = int(request.POST.get('Age'))
        cp = int(request.POST.get('cp'))
        Sex = int(request.POST.get('Sex'))
        thalach = int(request.POST.get('thalach'))
        Treshbps = int(request.POST.get('Treshbps'))
        exang = int(request.POST.get('exang'))
        Chol = int(request.POST.get('Chol'))
        oldpeak = int(request.POST.get('oldpeak'))
        Fbs = int(request.POST.get('Fbs'))
        ca = int(request.POST.get('ca'))
        Restecg = float(request.POST.get('Restecg'))
        thal = int(request.POST.get('thal'))
        slope = int(request.POST.get('slope'))
        res = heart_disease([Age,Sex,cp,Treshbps,Chol,Fbs,Restecg,thalach,exang,oldpeak,slope,ca,thal])
        context ={
            'data':res
        }
        return render(request,'heartDetection.html',context)
    return render(request,'heartDetection.html')

def self_examine(request):
    return render(request,'self_examine.html')
