# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:42:28 2017

@author: Xiaoxi
"""

%reset
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()
digits.images.shape
digits.images[3]
digits.target[3]

fig,axes = plt.subplots(10, 10, figsize=(8,8))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform = ax.transAxes, color = 'red')
    ax.set_xticks([])
    ax.set_yticks([])
    
    
'classification on digits, with logistic regression' 
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state = 2)  
                                                
Xtrain.shape
Xtest.shape                                                

'logistic regression'
clf = LogisticRegression(penalty = 'l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
accuracy_score(ytest, ypred) # but not tell where goes wrong
confusion_matrix(ytest, ypred)# evaluate the accuracy of classification

plt.imshow(np.log(confusion_matrix(ytest, ypred)),
           cmap = "Blues", interpolation = 'nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')           


fig,axes = plt.subplots(4, 4, figsize=(8,8))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform = ax.transAxes, 
            color = 'green' if (ytest[i] == ytest[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])






