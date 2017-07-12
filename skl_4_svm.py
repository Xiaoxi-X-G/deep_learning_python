# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:45:54 2017

@author: Xiaoxi
"""

%reset

import numpy as np
import matplotlib.pylab as plt

from scipy import stats

from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC # support vector classifier


import seaborn as sns; sns.set()

from mpl_toolkits import mplot3d

from IPython.html.widgets import interact


X, y = make_blobs(n_samples = 100, 
                  centers = 2,
                  random_state = 0,
                  cluster_std = 0.60)

plt.scatter(X[:,0], X[:,1], c=y, 
            s = 100,  # circle size
            cmap = 'spring'
            )

xfit = np.linspace(-1,3.5)


for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2,2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit-d, yfit+d,
                     edgecolor = 'none',
                     color = '#AAAAAA',
                     alpha = 0.4)
    
plt.xlim(-1, 3.5)    


'kernel'
clf = SVC(kernel = 'linear')
clf.fit(X, y)

'svm plot'

def plot_svc_decision_function(clf, ax = None):
    """ plot the decision fuction in for 2d
    """
    
    if ax is None:
        ax = plt.gca()
        
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)    
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)

    Y,X = np.meshgrid(y, x)
    
    P = np.zeros_like(X) #an array of zeros with the same shape and type as a given array
    
    for i, xi in enumerate(x): # index and value
        for j, yj in enumerate(y):
            P[i,j] = clf.decision_function([xi, yj])
            
    # plot the margins
    ax.contour(X,Y, P, colors = 'k',
               levels = [-1, 0, 1], alpha = 0.5,
                linestyles = ['--','-', '--'])
                

plt.scatter(X[:,0], X[:,1], c = y, s = 50, cmap = 'spring')
plot_svc_decision_function(clf)                
    
    
'circle datasets'
X, y = make_circles(50, factor = 0.1, noise = 0.1)
clf = SVC(kernel = 'linear').fit(X,y)
plt.scatter(X[:,0], X[:,1], c = y, s = 50, cmap = 'spring')
plot_svc_decision_function(clf)      

r = np.exp(-(X[:,0]**2 + X[:,1]**2))          
    
def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection = '3d')
    ax.scatter3D(X[:,0], X[:,1], r, c = y, s = 50, cmap = 'spring')
    ax.view_init(elev = elev, azim = azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

plot_3D()    

interact(plot_3D, elev = [-90, 90], azip = (-180, 180));

'change kernel to nonlinear'
# http://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel
clf = SVC(kernel = 'rbf') # radial basis function
clf.fit(X, y)

plt.scatter(X[:,0], X[:,1], c = y, s = 50, cmap = 'spring')
plot_svc_decision_function(clf)
