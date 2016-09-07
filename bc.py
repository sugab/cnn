# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:57:43 2016

@author: baguscahyono
"""

# Importing depedency lib
import numpy as np

# Base class for all layer
class Layer:

    learning_rate = 1

    def forward(self):
        raise NotImplementedError('Subclass must override forward function')

    def backward(self):
        raise NotImplementedError('Subclass must override backward function')

# Class implementation for Fully Connected Layer
class FCLayer(Layer):

    def __init__(self, dimension, activation=None, auto_update=False):
        # Initiate weight and bias based on given dimension
        self.weight = np.random.normal(0, 0.01, dimension)
        self.bias = np.ones(dimension[1])
        self.activation = activation
        self.auto_update = auto_update

    def forward(self, in_data):
        self.in_data = in_data
        net =  np.dot(self.in_data, self.weight) + self.bias

        # Check if we use activation in this layer
        if self.activation is None:
            self.out_data = net
        else:
            self.out_data = self.activation.forward(net)
            
        return self.out_data

    def backward(self, error):
        self.delta = error

        # Check if we use activation in this layer
        if self.activation is not None:
            self.delta *= self.activation.backward(self.out_data)

        next_error = self.delta.dot(self.weight.T)

        self.gradient()

        if self.auto_update:
            self.update()

        return (next_error, self.delta_w)

    def gradient(self):
        self.delta_w = self.in_data.T.dot(self.delta)
        return self.delta_w

    def update(self):
        self.weight -= self.delta_w * Layer.learning_rate
        self.bias -= np.sum(self.delta, axis=0) * Layer.learning_rate

# Class implementation for sigmoid layer
class SigmoidLayer:

    @staticmethod
    def forward(data):
        return 1/(1+np.exp(-data))

    @staticmethod
    def backward(data):
        return data*(1-data)

# Class implementation for Softmax layer
class SoftmaxLayer:

    @staticmethod
    def forward(data):
        exp_scores = np.exp(data)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def backward(data, label):
        data[range(len(label)), label] -= 1
        return data
