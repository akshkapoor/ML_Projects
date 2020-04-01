#In this Case Study we are predicting the breast cancer whether it is Malignant or Benign. Malignant means that the patient has the Cancer
and Benign means that he does not have it.
#We have been given the classic Breast cancer wisconsin (diagnostic) dataset available in Scikit-learn. We have features such as radius,
texture,perimeter,area etc of the cells taken from the Breast mass.
#We tried different classification models such as XGBoost,Artificial Neural Network(ANN) and SVM. As SVM trains on Support Vectors i.e. the
dataset points which can easily be fooled we achieved a mean accuracy score of 97% using K-Fold Cross Validation with cv=10.
#The technique can rapidly evaluate breast masses and classify them in an automated fashion.
