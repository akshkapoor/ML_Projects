# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:44:32 2020

@author: akshkapoor
"""

# Importing the libraries
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the training set
dataset = pd.read_csv('train.csv')
dataset.info()
dataset.describe().T

#Some EDA
sns.countplot(dataset['Survived'],hue=dataset['Sex'])
sns.countplot(dataset['Survived'],hue=dataset['Embarked'])
sns.countplot(dataset['Survived'],hue=dataset['Pclass'])
sns.distplot(dataset['Age'].dropna())
dataset['Fare'].plot.hist()
sns.countplot(dataset['SibSp'])

#Importing the test set
test_set=pd.read_csv('test.csv')
test_set.info()

dataset=pd.concat([dataset,test_set],axis=0,ignore_index=True)


#Feature Extraction from Name Variable
def split(x):
    y=x.split(',')[1]
    return y.split(' ')[1]

dataset['Name']=dataset['Name'].apply(split)

dataset.loc[dataset.Name=='the','Name']='Countess.'
#ss=list(set(dataset['Name']))
dataset['Name']=dataset['Name'].apply(lambda x:x.replace('.',''))


Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royal",
                        "Don":        "Royal",
                        "Sir" :       "Royal",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "Countess":   "Royal",
                        "Dona":       "Royal",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royal"

                        }

def titlemap(x):
    return Title_Dictionary[x]

dataset['Name']=dataset['Name'].apply(titlemap)

def isRare(title):
    if title == "Mr" or title == "Mrs" or title == "Master" or title == "Miss":
        return 0
    return 1

dataset['Name']=dataset['Name'].apply(isRare)

#Dropping PassengerId and Ticket as they are not adding any meaning
df=dataset.drop(['PassengerId','Ticket'],axis=1)

df.info()

df.describe()

#Feature Extraction from Cabin 1 if a peron has cabin 0 if not
def cabin_chk(x):

    if x!=0:
        return 1
    else:
        return 0
    
df['Cabin']=df['Cabin'].fillna(0)
df['Cabin_al']=df['Cabin'].apply(cabin_chk)

df=df.drop('Cabin',axis=1)

#Imputing missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median',axis=0)
imputer1=Imputer(strategy='median',axis=0)
df['Age']=imputer.fit_transform(df.iloc[:,0:1])
df['Fare']=imputer.fit_transform(df.iloc[:,2:3])

df.loc[pd.isna(df.Embarked),'Embarked']='S'

df['Sex']=df['Sex'].apply(lambda x: 0 if x=='male' else 1)

#Some more EDA
df.corrwith(df.Survived).plot.bar()

sns.heatmap(df.corr(),annot=True)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
df['Embarked']=encoder.fit_transform(df['Embarked'])

X=df.iloc[:,[0,1,2,3,4,5,6,7,9]].values
y=df.iloc[:,8].values

oh=OneHotEncoder(categorical_features=[1])
X=oh.fit_transform(X).toarray()

X_testset=X[891:,1:]
X=X[:891,1:]
y=y[:891]

#Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#X=sc.fit_transform(X)
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_testset=sc.transform(X_testset)

#Fitting the model to training set
'''from xgboost import XGBClassifier
classifier=XGBClassifier(gamma=0.1,learning_rate=0.1,n_estimators=150)
classifier.fit(X_train,y_train)'''

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',C=1,gamma=0.1,random_state = 0)
classifier.fit(X_train, y_train)

#Tuning the hyperparameters using Grid Search
parameters=[{'learning_rate':[0.1,0.01,1],'n_estimators':[100,150,200],'gamma':[0.1,0.01,1]}]
parameters_svm=[{'kernel':['linear'],'C':[1,10,100],'gamma':[0.1,0.01,1]},
                 {'kernel':['rbf'],'C':[1,10,100],'gamma':[0.1,0.01,1]}]

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=classifier,param_grid=parameters_svm,scoring='accuracy',cv=10)
grid=grid.fit(X_train,y_train)

best_score=grid.best_score_
best_parms=grid.best_params_

#Predicting the test-set results
y_pred=classifier.predict(X_test)

#Calculating the accuracy profile
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)

asc=accuracy_score(y_test,y_pred)

cr=classification_report(y_test,y_pred)

#Predicting the Results
y_testset=classifier.predict(X_testset).astype(int)

#Writing to the submission File
result_var={'PassengerId':test_set['PassengerId'],'Survived':y_testset}

pd.DataFrame(result_var).to_csv('Results.csv',index=False)



###############################################################################
'''dataset.corrwith(dataset['Survived'])

vals=np.size(dataset['SibSp'].unique())
plt.hist(dataset['SibSp'],bins=vals)

plt.show()


dataset[dataset['SibSp']==1]['Survived'].value_counts()
dataset[dataset['SibSp']==2]['Survived'].value_counts()
dataset[dataset['SibSp']==3]['Survived'].value_counts()
dataset[dataset['SibSp']==5]['Survived'].value_counts()###
dataset[dataset['SibSp']==8]['Survived'].value_counts()###

dataset[dataset['Parch']==3]['Survived'].value_counts()
dataset[dataset['Parch']==4]['Survived'].value_counts()###
dataset[dataset['Parch']==5]['Survived'].value_counts()
dataset[dataset['Parch']==6]['Survived'].value_counts()###'''

