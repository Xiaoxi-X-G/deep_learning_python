# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:29:29 2018

@author: Xiaoxi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import data
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values  #pd.df -> ndnumpy
y = dataset.iloc[:, 13].values

# Encode categorical var
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1]) # index indicate cate var loc
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #remove the correlated one col from country column after onehot encoder


# split training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, 
                                                 random_state = 0)

# feature scaling: import to ann
# this step goes after split test and train, as otherwise, the training feature may not cover entire range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #removing the mean and scaling to unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# import Tenser flow
from keras.models import Sequential
from keras.layers import Dense # used to random init the weights to small numbers close to 0, choose activation function, etc. ctrl+i for help

# choose rectifier activation function for hidden layer to:
#    - reduce vanishing gradient problem caused by sigmoid function
# choose sigmoid activation function for output layer due to:
 #   - output a probabality     


#######################################################
#1.  without k-fold cross validation
# init ann: define as a sequence of layers
classifier = Sequential()

# add input layer and hidden layers
# there is no rule of thumb of number of nodes in hidder layers
# empiricially, it equals  (input nodes + output nodes) / 2
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu',
                     input_dim = 11))

# add the second hidden layer
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu'
                     ))

# add output layer
classifier.add(Dense(output_dim = 1,
                     init = 'uniform',
                     activation = 'sigmoid'
                     ))

# compile
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# making the prediction and evaluation the model
classifier.fit(x = X_train, 
               y = y_train, 
               batch_size = 10,
               nb_epoch = 100)

# prediction on test set
y_pred_pro = classifier.predict(X_test)
y_pred = (y_pred_pro > 0.5)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


###################################################################
#2. use k-cross validation ???
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    # add input layer and hidden layers
    # there is no rule of thumb of number of nodes in hidder layers
    # empiricially, it equals  (input nodes + output nodes) / 2
    classifier.add(Dense(output_dim = 6,
                         init = 'uniform',
                         activation = 'relu',
                         input_dim = 11))
    
    # add the second hidden layer
    classifier.add(Dense(output_dim = 6,
                         init = 'uniform',
                         activation = 'relu'
                         ))
    
    # add output layer
    classifier.add(Dense(output_dim = 1,
                         init = 'uniform',
                         activation = 'sigmoid'
                         ))
    
    # compile
    classifier.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10,
                             nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1 # use all cpu
                             )
# avg accurate
mean = accuracies.mean()
variance = accuracies.std()

###########################################################
# imporve ann

# dropoff regularization:
#    - to reduce overfitting 
#    - random disconnect few neuros each time
#    - used when high accuracy in test but low in test set
#    - used when high variance
#    - start with 10% dropout, if it dones't solve the overfitting, increase the rate

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential

#