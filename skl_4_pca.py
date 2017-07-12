# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:31:48 2017

@author: Xiaoxi
"""
%reset

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats, linalg

from sklearn.decomposition import PCA

import seaborn as sns; sns.set()


np.random.seed(1)
X = np.dot(np.random.random(size = (2,2)),
           np.random.normal(size = (2,200))).T
           
plt.plot(X[:,0], X[:,1], 'o')       
plt.axis('equal')    

pca = PCA(n_components = 2)
pca.fit(X)

pca.explained_variance_ratio_ # eigen value ratio
pca_component = pca.components_ # signal space in svd

np.dot(pca_component[1,], pca_component[0,])

'svd'
U, s, Vh = linalg.svd(X)
np.diag(s)
np.dot(Vh[1,], Vh[0,])
s**2/sum(s**2) # same as pca.explained_variance_ratio_ 


'plot - svd'
plt.plot(X[:,0],X[:,1], 'o', alpha = 0.5)
for length, vector in zip(s, Vh):
    v = vector*np.sqrt(length)
    print(length, vector, v)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw = 3)
plt.axis('equal')    


'plot - pca'
plt.plot(X[:,0],X[:,1], 'o', alpha = 0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector*np.sqrt(length)
    print(length, vector, v)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw = 3)
plt.axis('equal')  
    
    
'95% of variace'
clf = PCA(n_components = 1) 
X_trans = clf.fit_transform(X)   #apply the dimensionality reduction   

X_svd_result = np.dot(X, Vh.T) # un rotate by V
X_trans_svd = X_svd_result[:,0].reshape(X_trans.shape) # same as X_trans

'pca - result'
X_new = clf.inverse_transform(X_trans)  # ? Transform data back to its original space.
plt.plot(X[:,0],X[:,1], 'o', alpha = 0.5) #
plt.plot(X_new[:,0],X_new[:,1], 'ob', alpha = 0.5)


'svd - result'
X_new = clf.inverse_transform(X_trans_svd)  # ? Transform data back to its original space.
plt.plot(X[:,0],X[:,1], 'o', alpha = 0.5) 
plt.plot(X_new[:,0],X_new[:,1], 'ob', alpha = 0.5)