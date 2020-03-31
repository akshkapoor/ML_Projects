# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 02:07:55 2020

@author: akshkapoor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import linalg as LA

#Importing the dataset
dataset = pd.read_csv('churn_data.csv')
dataset.info()

#Plotting the histograms
dataset2=dataset.drop(['user','churn'],axis=1)
plt.figure(figsize=(15,12))
plt.suptitle('Histograms',fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,5,i)
    f=plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    vals=np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins=vals)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


## Pie Plots
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    values=dataset2.iloc[:,i-1].value_counts(normalize=True).values
    index=dataset2.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

##Checking the distribution of uneven features
dataset[dataset['waiting_4_loan']==1]['churn'].value_counts()
dataset[dataset['rejected_loan']==1]['churn'].value_counts()
dataset[dataset['left_for_one_month']==1]['churn'].value_counts()
    
##Checking the correlations with heatmap
plt.figure(figsize=(27,27))
sns.heatmap(round(dataset.corr(),1),annot=True)

##Checking the correlations with response variable
dataset.corrwith(dataset.churn).plot.bar()

#Checking No of Null values in columns
sum(dataset.credit_score.isna())/27000
sum(dataset.rewards_earned.isna())/27000

dataset['credit_score'].mean()
dataset['credit_score'].median()

#Saving the correlations to a DataFrame
df=round(dataset.corr(),1)

dataset.info()

#Analyzing the payment-type
len(dataset[dataset.payment_type=='na']['user'])/27000

#Dropping the columns with NULLS and Maximum Correlations
dataset=dataset.drop(['housing','user','app_web_user','rewards_earned','credit_score','deposits'
                      ,'android_user'],axis=1)

dataset.info()

#Setting the na values to mode of Data
dataset.loc[dataset.payment_type=='na','payment_type']='Bi-Weekly'
dataset.loc[dataset.payment_type=='na','payment_type']

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
le1=LabelEncoder()
dataset['payment_type']=le.fit_transform(dataset['payment_type'])
dataset['zodiac_sign']=le1.fit_transform(dataset['zodiac_sign'])

#Seggregating the features and response variable
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values


#Imputing the age with mode
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
X[:,0:1]=imputer.fit_transform(X[:,0:1])
'''imputer1=Imputer(missing_values='NaN',strategy='median',axis=0)
X[:,1:2]=imputer1.fit_transform(X[:,1:2])
imputer2=Imputer(missing_values='NaN',strategy='median',axis=0)
X[:,23:24]=imputer1.fit_transform(X[:,23:24])'''

#Check if any column contains null values
pd.DataFrame(X).info()

#Checking multi-colllinearity
LA.matrix_rank(X)

#One-hot encoding of categorical features
oh=OneHotEncoder(categorical_features=[13])
X=oh.fit_transform(X).toarray()
X=X[:,1:]

oh1=OneHotEncoder(categorical_features=[20])
X=oh1.fit_transform(X).toarray()
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
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=150,criterion='entropy')
classifier.fit(X_train,y_train)

'''from xgboost import XGBClassifier
classifier=XGBClassifier(gamma=0.1,learning_rate=0.1,n_estimators=200)
classifier.fit(X_train,y_train)'''

y_pred=classifier.predict(X_test)

'''#Tuning the hyperparameters using Grid Search
parameters=[{'learning_rate':[0.1,0.01,1],'n_estimators':[100,150,200],'gamma':[0.1,0.01,1]}]

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid=grid.fit(X_train,y_train)

best_score=grid.best_score_
best_parms=grid.best_params_'''

#Calculating the accuracy profile
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)

asc=accuracy_score(y_test,y_pred)

cr=classification_report(y_test,y_pred)
