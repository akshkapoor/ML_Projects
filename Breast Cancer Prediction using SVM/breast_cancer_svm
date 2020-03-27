# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:56:55 2020

@author: akshkapoor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

#Building the dataset
X=dataset['data']
y=dataset['target']
X_df=pd.DataFrame(data=X,columns=dataset['feature_names'])
y_df=pd.DataFrame(data=y,columns=['target'])
dataset=pd.concat([X_df,y_df],axis=1)

#Assigning features and response
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
classifier.fit(X_train, y_train)

'''from xgboost import XGBClassifier
classifier=XGBClassifier(random_state=0)
classifier.fit(X_train,y_train)'''

#Applying Neural Networks
'''from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=16,activation='relu',input_dim=30,init='uniform'))
classifier.add(Dense(units=16,activation='relu',init='uniform'))
classifier.add(Dense(units=1,activation='sigmoid',init='uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=75)'''

#Tuning the hyperparameters using Grid Search
parameters=[{'kernel':['linear'],'C':[1,10,100]},{'kernel':['rbf'],'C':[1,10,100]}]

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid=grid.fit(X_train,y_train)

best_score=grid.best_score_
best_parms=grid.best_params_

#Predicting the response variable
y_pred=classifier.predict(X_test)

'''def prob(x):
    if x>=0.8:
        return 1
    else:
        return 0

y_pred=[prob(x) for x in y_pred]'''

#Checking the accuracy profile
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)

asc=accuracy_score(y_test,y_pred)

cr=classification_report(y_test,y_pred)
