# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:06:12 2020

@author: akshkapoor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('new_appdata10.csv')

dataset.info()

#Converting object to Datetime Datatype
dataset['first_open']=pd.to_datetime(dataset['first_open'])
dataset['enrolled_date']=pd.to_datetime(dataset['enrolled_date'])

#Feature Engineering


dataset['Difference']=(dataset['enrolled_date']-dataset['first_open']).astype('timedelta64[h]')

dataset['Difference'].describe()

plt.hist(dataset['Difference'].dropna(),range=[0,100])
plt.show()

dataset.loc[dataset.Difference>48,'enrolled']=0

plt.hist(dataset[dataset.enrolled==1]['dayofweek'])
plt.show()

dataset['hour']=dataset['hour'].apply(lambda x:x[:2])
dataset['hour']=dataset['hour'].astype(int)

dataset.info()

#Importing the list of top screens
dsc=pd.read_csv('top_screens.csv').top_screens

#Funnels
savings_screens=['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7',
                 'Saving8','Saving9','Saving10']

cm_screens=['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']

cc_screens=['CC1','CC1Category','CC3']

loan_screens=['Loan','Loan2','Loan3','Loan4']

def sav_counts(x):
    
    cnt_sav=len([i for i in x.split(',') if i in savings_screens])
    #cnt_cm=len([i for i in x.split(',') if i in cm_screens])
    #cnt_cc=len([i for i in x.split(',') if i in cc_screens])
    #cnt_loan=len([i for i in x.split(',') if i in loan_screens])
    
    return cnt_sav

def cm_counts(x):
    
    #cnt_sav=len([i for i in x.split(',') if i in savings_screens])
    cnt_cm=len([i for i in x.split(',') if i in cm_screens])
    #cnt_cc=len([i for i in x.split(',') if i in cc_screens])
    #cnt_loan=len([i for i in x.split(',') if i in loan_screens])
    
    return cnt_cm

def cc_counts(x):
    
    #cnt_sav=len([i for i in x.split(',') if i in savings_screens])
    #cnt_cm=len([i for i in x.split(',') if i in cm_screens])
    cnt_cc=len([i for i in x.split(',') if i in cc_screens])
    #cnt_loan=len([i for i in x.split(',') if i in loan_screens])
    
    return cnt_cc

def loan_counts(x):
    
    #cnt_sav=len([i for i in x.split(',') if i in savings_screens])
    #cnt_cm=len([i for i in x.split(',') if i in cm_screens])
    #cnt_cc=len([i for i in x.split(',') if i in cc_screens])
    cnt_loan=len([i for i in x.split(',') if i in loan_screens])
    
    return cnt_loan

def other_counts(x):
    
    cnt_oth=len([i for i in x.split(',') if i not in savings_screens and i not in cm_screens 
                 and i not in cc_screens and i not in loan_screens])
    
    return cnt_oth

dataset['SavingsCount']=dataset['screen_list'].apply(sav_counts)
dataset['CmCount']=dataset['screen_list'].apply(cm_counts)
dataset['CcCount']=dataset['screen_list'].apply(cc_counts)
dataset['LoanCount']=dataset['screen_list'].apply(loan_counts)
dataset['OtherCount']=dataset['screen_list'].apply(other_counts)

#Seggregating the features and response
X=dataset.iloc[:,[2,3,4,6,7,8,11,13,14,15,16,17]].values
y=dataset.iloc[:,9].values

#Splitting the data into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from xgboost import XGBClassifier
classifier=XGBClassifier(random_state=0)
classifier.fit(X_train,y_train)

# Fitting SVM to the Training set
'''from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
classifier.fit(X_train, y_train)'''

y_pred=classifier.predict(X_test)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
cvs.mean()

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)

asc=accuracy_score(y_test,y_pred)

cr=classification_report(y_test,y_pred)
