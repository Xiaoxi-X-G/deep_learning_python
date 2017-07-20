# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:34:28 2017

@author: Xiaoxi
"""

# http://neuralnetworksanddeeplearning.com/chap1.html

import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

class Network(object):    
    def __int__(self, sizes):
        # sizes contains the number of neurons
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        """ Return the output of the network if 'a' is input."""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
        
         
        
                                
                                
                            
        