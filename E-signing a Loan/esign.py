# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:36:26 2020

@author: akshkapoor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('P39-Financial-Data.csv')
dataset.info()

##Checking the distribution of uneven features
dataset[dataset['has_debt']==0]['e_signed'].value_counts()
dataset[dataset['pay_schedule']=='monthly']['e_signed'].value_counts()
dataset[dataset['pay_schedule']=='semi-monthly']['e_signed'].value_counts()

##Plotting the histograms
dataset2=dataset.drop(['entry_id','e_signed'],axis=1)

plt.figure(figsize=(15,12))
plt.suptitle('Histograms',fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,5,i)
    f=plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    vals=np.size(dataset2.iloc[:,i-1].unique())
    if vals>=100:
        vals=100
    plt.hist(dataset2.iloc[:,i-1],bins=vals)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

##Plotting the Correlations with Response Variable
dataset.corrwith(dataset['e_signed']).plot.bar()

##Cleaning the data based on EDA
dataset=dataset.drop('months_employed',axis=1)
dataset['personal_account_y']=round(dataset['personal_account_y']+dataset['personal_account_m']/12,2)
dataset=dataset.drop('personal_account_m',axis=1)

##Some more EDA
df=dataset.describe()
dataset[dataset['years_employed']<=3]['e_signed'].value_counts()

df_corr=dataset.corr()
sns.heatmap(df_corr,annot=True)

dataset.info()

##Dropping the index key column
dataset=dataset.drop('entry_id',axis=1)

##Label Encoding the Categorical Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
dataset['pay_schedule']=encoder.fit_transform(dataset['pay_schedule'])

##Separating the data into Features and Response
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

oh=OneHotEncoder(categorical_features=[1])
X=oh.fit_transform(X).toarray()
X=X[:,1:]

#Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting the model to training set
from xgboost import XGBClassifier
classifier=XGBClassifier(gamma=0.1,learning_rate=0.1,n_estimators=200)
classifier.fit(X_train,y_train)

#Predicting the test-set results
y_pred=classifier.predict(X_test)

#Calculating the accuracy profile
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)

asc=accuracy_score(y_test,y_pred)

cr=classification_report(y_test,y_pred)
