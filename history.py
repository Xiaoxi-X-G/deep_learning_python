import matplotlib.pyplot as plt
import numpy as np
#import random




class Network(object):
    
    def __init__(self, sizes):
        """The list size contains the number of neurons in the
        respectively layers of the network
        
        The biases and weights for the network are initalized
        randomly, with N(0, 1)"""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biaes = [np.random.randn(y,1) for y in sizes[1:]]
        self.weight = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    #@staticmethod
    def sigmod(z):
        return 1.0/(1.0+np.exp(-z))
        
    def feedforward(self, a):
        """return the output of the network if "a" is input
        """
        for b,w in zip(self.biaes, self.weight):
            #a = Network.sigmoid(np.dot(w, a)+b)
            a = self.sigmoid(np.dot(w, a)+b)
        return a
        
        
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None):
                """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
       
       if test_data: 
           n_test = len(test_data)
            
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j) 
                
                
                
                           
                           
                            
                            
                            
                            
                            