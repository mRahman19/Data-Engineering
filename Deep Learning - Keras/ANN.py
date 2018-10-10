# Dataset - 
# dataset contain last 6 month details of customer of Bank 
# Our job is to figure out which customer has high probability to leave the bank.
# 0 define stay in the Bank and 1 define leave the Bank. 
# This method can use any other scenario not only for Bank. for instance - should the person get loan or not, should the person get credit card or not. 

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Convert categorical features
sex = pd.get_dummies(dataset['Gender'], drop_first=True)
geography = pd.get_dummies(dataset['Geography'], drop_first=True)

#Remove unnecessery column and add categorical column
dropData = pd.concat([dataset, sex, geography], axis=1 )
finalData = dropData.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'], axis=1)

#Create X and y of independent and dependend variable
X = finalData.drop('Exited', axis=1)
y = finalData['Exited']

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Import libraries for ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from.keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

#Initialize ANN 
classifier = Sequential()
#Adding input layer and create first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fit ANN to training data set with 10 batch and train 100 times
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#Making prediction and use threshhold to separate predicted value into 0 and 1
prediciton = classifier.predict(X_test)
prediction = (prediciton > .5)

#making the confusion and classification matrix
cm = confusion_matrix(y_test, prediction)
cr = classification_report(y_test, prediciton)
# Result = CM - .8625

# Predict single new observation to see how our model predict - 
newPrediction = classifier.predict(scaler.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
newPrediction = (newPrediction > 0.5)
#Result = False

#-------------------------------------------------------------------------------------------#

# Evaluating and Improving ANN 
# Build ANN architecture by using funciton call build_classifier and pass this function to Keras classifier
#Use dropout to prevent model form overfitting, use dropout 10%
def build_classifier():

    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p=0.1))
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# K fold cross validation use to evaluate model performance and get correct accuracy rate on training and test set that reduce high bias and high variance of model
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()
#Result = Accuracy - .844 and Variance - 0.02, So we can say our model is low variance and low bias

#-------------------------------------------------------------------------------------------#

#ANN Tuning, 
# Use gridSearch to find optimum parameter to improve model performance
# Create dictionary that will contain hyper parameter to find best values. Create optimizer dict and pass it into build.classifier function. 

def build_classifier(optimizer):

    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p=0.1))
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'optimizer', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameter = {'batch_size': [25,32],
             'nb_epoch': [100,500],
             'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameter,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Result = 
# Best parameter - Batch size=25, nb_epoch=500, optimizer = rmsprop
# Best accuracy = .85












