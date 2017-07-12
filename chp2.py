# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:08:31 2017

@author: Xiaoxi
"""

%reset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab 


path = "C:/Users/Xiaoxi/Dropbox/work/2017/Others/python"

df = pd.read_csv(path + "/data/iris.csv", header = None)

df.describe()
df.dtypes
df.shape
df.head(n = 10)
df.tail(n = 10)

y = df.iloc[0:100, 4].values # to numpy.ndarry

Y = np.where(y== 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values # to numpy.ndarry

plt.figure()
plt.ion() 
plt.scatter(X[:50, 0], X[:50, 1],
            color = 'red', marker = 'o', label = 'setosa')

plt.scatter(X[50:100, 0], X[50:100, 1],
            color = 'blue', marker = 'x', label = 'virginica')
            
plt.xlabel('petal length')      
plt.ylabel('sepal length')            

plt.legend(loc = 'upper left')      

plt.show()