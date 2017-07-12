# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:29:40 2017

@author: Xiaoxi
"""

%reset

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import seaborn; seaborn.set() # set seaborn plot defaults
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier #stochastic gradient descent (SGD) learning
from sklearn.datasets.samples_generator import make_blobs

from sklearn.linear_model import LinearRegression

from sklearn import neighbors, datasets

from sklearn.ensemble import RandomForestRegressor


model = LinearRegression(normalize = True)
print(model)
x = np.arange(10)
y = 2*x + 1

plt.plot(x,y,'o')

# input data for sklearn is 2-D + feature
X = x[:, np.newaxis]
model.fit(X,y)

model.coef_
model.intercept_
model.residues_


'supervised learning'
boston = datasets.load_boston()

iris = datasets.load_iris()

X,y = iris.data, iris.target

X.shape
y.shape

# create a model
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)

# fit a model
knn.fit(X, y)

# what kind of iris has 3m X 5cm sepal and 4cm X 2cm petal
# call the prediction
result = knn.predict([[3,5,4,2],])
iris.target_names[result]


'regression'
np.random.seed(0)
X = np.random.random(size = (20,1)) 
y = 3*X.squeeze()+ 2 + np.random.rand(20) # change 1 dim matrix to arrary

plt.plot(X.squeeze(), y, 'o')

model = LinearRegression()
model.fit(X, y) # X is a matrix format

# plot
X_fit = np.linspace(0,1,100)[:, np.newaxis] # add another dim 
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, "o")
plt.plot(X_fit.squeeze(), y_fit)

'random forest'
model = RandomForestRegressor(n_estimators = 20, # number of trees
                              max_depth = 1) # depth of tree
                              
model.fit(X, y)                              
# plot
X_fit = np.linspace(0,1,100)[:, np.newaxis] # add another dim 
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, "o")
plt.plot(X_fit.squeeze(), y_fit)


'optical character recognition'
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