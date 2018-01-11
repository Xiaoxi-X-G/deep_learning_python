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
training_data_old, test_data_old = mnist.load_data()

training_data = list(zip(training_data_old[0], training_data_old[1]))
test_data = list(zip(test_data_old[0], test_data_old[1]))


def print_img(dat, ind):
    if type(dat[ind]) is not tuple:
        print("Not a tuple type")
        return 1
    elif type(dat[ind]) is tuple:
        #plt.imshow(dat[0][ind])
        plt.imshow(dat[ind][0])
        plt.title("Number: " + str(dat[ind][1]))
        
        
#____________________________________


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
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
#            rnd_ind = np.arange(training_data[0].shape[0])
#            random.shuffle(rnd_ind)
#            training_data = (training_data[0][rnd_ind], training_data[1][rnd_ind])
            random.shuffle(training_data)
            
            # syntex
            mini_batches = [
                    training_data[k: k+mini_batch_size]
                    for k in range(0, n , mini_batch_size) # (start, stop, interval)
                    ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test)
                      )
            else:
                print( "Epoch {0} complete".format(j)   )             
            
        
    # update mini_batch
    def update_mini_batch(self, mini_batch, eta):
        """
        Using gradient descent to update the network's weight and biases on each
        mini_batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backdrop(x.reshape(784,1), y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nb+dnw for nb, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b -(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    #
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs
        the correct results
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x,y) in test_data]   
        return sum(int(x==y) for (x,y) in test_results)
         
    
    #
    def backdrop(self, x, y):
        """
        Return a tuple ''(nabla_b, nabla_w)'' representing the gradient for the 
        cost function C_x. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #feedforward
        activation = x
        activations = [x] # a list to store all activation, layer by layer
        zs = [] # list to store all the z vectors, layers by layers
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backward pass
        delta = self.cost_derivative(activations[-1], y)* \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
                            
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of prtial derivatives \partial C_x / partial a
        for the output activations
        """
        return (output_activations-y)