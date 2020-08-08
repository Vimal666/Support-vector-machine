# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:30:13 2020

@author: Vimal PM
"""
#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC
#loading the datastet
train=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Work done\SVM\SalaryData_Train.csv")
Train=train
test=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Work done\SVM\SalaryData_Test.csv")
Test=test
train.columns
#Index(['age', 'workclass', 'education', 'educationno', 'maritalstatus',
  #     'occupation', 'relationship', 'race', 'sex', 'capitalgain',
   #    'capitalloss', 'hoursperweek', 'native', 'Salary']
   
#coverting categorical to numerical from train dataset
Le=preprocessing.LabelEncoder()   
Train["workclass"]=Le.fit_transform(Train["workclass"])
Train["education"]=Le.fit_transform(Train["education"])
Train["maritalstatus"]=Le.fit_transform(Train["maritalstatus"])
Train["occupation"]=Le.fit_transform(Train["occupation"])
Train["relationship"]=Le.fit_transform(Train["relationship"])
Train["race"]=Le.fit_transform(Train["race"])
Train["sex"]=Le.fit_transform(Train["sex"])
Train["native"]=Le.fit_transform(Train["native"])


#coverting categorical to numerical from test dataset
Test["workclass"]=Le.fit_transform(test["workclass"])
Test["education"]=Le.fit_transform(test["education"])
Test["maritalstatus"]=Le.fit_transform(test["maritalstatus"])
Test["occupation"]=Le.fit_transform(test["occupation"])
Test["relationship"]=Le.fit_transform(test["relationship"])
Test["race"]=Le.fit_transform(test["race"])
Test["sex"]=Le.fit_transform(test["sex"])
Test["education"]=Le.fit_transform(test["native"])
Test["native"]=Le.fit_transform(test["native"])

X_train=Train.iloc[:,0:13]
Y_train=Train.iloc[:,13]
X_test=Test.iloc[:,0:13]
Y_test=Test.iloc[:,13]
#building the svm model using linear trick
svm_linear=SVC(kernel="linear")
svm_linear.fit(X_train,Y_train)
pred_linear=svm_linear.predict(X_test)
#accuracy
np.mean(pred_linear==Y_test)
pd.crosstab(Y_test,pred_linear)

#building second model using  poly kernel
svm_poly=SVC(kernel="poly")
svm_poly.fit(X_train,Y_train)
pred_poly=svm_poly.predict(X_test)
#accuracy
np.mean(pred_poly==Y_test)
pd.crosstab(Y_test,pred_poly)
#building 3rd model using rbf kernel
svm_rbf=SVC(kernel="rbf")
svm_rbf.fit(X_train,Y_train)
pred_rbf=svm_rbf.predict(X_test)
#accuracy
np.mean(pred_rbf==Y_test)
pd.crosstab(Y_test,pred_rbf)
