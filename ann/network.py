# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:34:28 2017

@author: Xiaoxi
"""

# http://neuralnetworksanddeeplearning.com/chap1.html

import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
import random
# load pre-shuffled mnist data into train and test
#(X_train, y_train),(X_test, y_test) = 
training_data, test_data = mnist.load_data()


def print_img(dat, ind):
    if type(dat) is not tuple:
        print("Not a tuple type")
        return 0
    elif type(dat) is tuple:
        plt.imshow(dat[0][ind])
        plt.title("Number: " + str(dat[1][ind]))
        
        
#____________________________________


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


#_____________________________________


class Network(object):
    def __init__(self, sizes):
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
        
    
    def backprop(self, x, y):
        """
        Return a tuple ''(nabla_b, nabla_w)'' representing the
        gradient for the cost function C_x. ''nabla_a'' and 
        ''nabla_w'' are layer-by-layer lists of numpy arrays,
        similar to ''self.biases'' and '''self.weights''
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
    # stochastic gradient descent    
    def SGD(self, training_data, epochs, mini_batch_size, 
            eta, test_data=None):
        """ 
        Train ann using mini-batch stochastic gradient descent.
        The 'training_data' is a list of tuples '(x, y)' representing
        the training inputs and the desired outputs. 
        
        The other non-optional parameters are self-explanatory. 
        
        If 'test-data' is provided then the network will be evaluated
        against the test data after epoch, and partial progress printed
        out. This is useful for tracking progress, but slows things
        down substantially.
        
        ______ similar as updating Lagrangian multiplier _____
        sdg: update gradient by 1 additional sample each incremental
        batch gradient descent: use all 'm' examples in each incremental
        mini-batch gradient descent: only use b examples in each incremental 
             
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs): # 0 - epochs
            random.shuffle(training_data)
        
            
            
            
            
        
        
         
        
                                
                                
                            
        