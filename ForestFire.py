# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:56:50 2020

@author: Vimal PM

"""
#importing neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

#opening my dataset using pd.read_csv
forest=pd.read_csv("D:\\DATA SCIENCE\\ASSIGNMENT\\Work done\SVM\\forestfires.csv")
forest.columns
#Index(['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
  #     'rain', 'area', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
  #     'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
  #     'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
  #     'monthoct', 'monthsep', 'size_category']
Forest=forest
#converting my categorical datas to numerical data
Le=preprocessing.LabelEncoder()
Forest["month"]=Le.fit_transform(forest["month"])
Forest["day"]=Le.fit_transform(forest["day"])
#checking the missing values
Forest.isna().sum() #no missing values
#getting the summary scores
Forest.describe()
#visualizing the variables using box plot
plt.boxplot(Forest["FFMC"]);plt.ylabel(Forest["month"])
sns.boxplot(x="month",y="size_category",data=Forest,palette = "hls")
sns.boxplot(x="size_category",y="day",data=Forest,palette="hls")
X=Forest.iloc[:,0:30]
Y=Forest.iloc[:,30]
#spliting my data's into train test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
#building the svm model and fitting into my train datasets
svm_linear=SVC(kernel="linear")
svm_linear.fit(X_train,Y_train)
pred_linear=svm_linear.predict(X_test)
#accuracy
np.mean(pred_linear==Y_test)
#0.9903846153846154(99%)
pd.crosstab(Y_test,pred_linear)
#model building using poly
svm_poly=SVC(kernel="poly")
svm_poly.fit(X_train,Y_train)
pred_poly=svm_poly.predict(X_test)
#acccuracy
np.mean(pred_poly==Y_test)
#0.9711538461538461(97%)
pd.crosstab(Y_test,pred_poly)

#building 3rd model using RBF
svm_rbf=SVC(kernel="rbf",gamma='scale')
svm_rbf.fit(X_train,Y_train)
pred_rbf=svm_rbf.predict(X_test)
np.mean(pred_rbf==Y_test)
# 0.7403846153846154(74%)
pd.crosstab(Y_test,pred_rbf)

##From my analysis I can say linear is the best kernel trick to classify my data's more accurately which giving around 99%
