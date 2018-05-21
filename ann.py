# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:33:44 2018

@author: Asad
"""

#Part-1: Data Pre-proccessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
#This is to get rid of the dummy vairable trap 
#by converting catagories into numerical values and then avearging them, sort of.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Used for easing computational load
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part-2: Making the ANN
# Importing Keras Library
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# Keras latest API - Dense(activation="relu", input_dim=11, (ouput)units=6, kernel_initializer="uniform")
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
#classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer with dropout
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
#classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit( X_train, y_train, batch_size = 25, epochs = 500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# homework - test = [600, 'France', 'male', 40, 3, 60000, 2, 1, 1, 50000]
new_pred = classifier.predict(sc.transform(np.array([[ 0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, improving, and tunning the ANN

# Evalutating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_Classifier() :
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier( build_fn = build_Classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
varience = accuracies.std()

# Improving the ANN
# Dropout regularization to reduce overfitting if needed


# Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_Classifier(optimizer) :
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier( build_fn = build_Classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit( X_train, y_train)
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_
