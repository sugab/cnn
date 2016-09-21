# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:57:43 2016

@author: baguscahyono
"""

# Importing depedency lib
import numpy as np

#%% BASE LAYER
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

#%% ACTIVATION LAYER
# Class implementation for ReLU layer
class ReLULayer:
    def forward(self, data):
        return np.maximum(0, data)

    def backward(self, data):
        return (data > 0) * 1

# Class implementation for ELU layer
class ELULayer:
    def forward(self, data, alpha=0.1):
        return np.maximum(alpha * np.exp(data) - 1, data)

    def backward(self, data, alpha=0.1):
        return np.where(data > 0, 1, data + alpha)

# Class implementation for sigmoid layer
class SigmoidLayer:
    def forward(self, data):
        return 1 / (1 + np.exp(-data))

    def backward(self, data):
        return data * (1 - data)

# Class implementation for Tanh layer
class TanhLayer:
    def forward(self, data):
        return 2 / (1 + np.exp(-2 * data)) - 1

    def backward(self, data):
        return 1 - (data ** 2)

# Class implementation for Softmax layer
class SoftmaxLayer:
    def forward(self, scores, target):
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        corect_logprobs = -np.log(probs[range(len(target)), target])
        data_loss = np.sum(corect_logprobs)

        return probs, data_loss

    def backward(self, scores, target):
        scores[range(len(target)), target] -= 1
        return scores

#%% NORMALIZATION
class BatchNorm:
    def forward(x, gm, bt, eps):
        N = x.shape[0]

        mu = 1. / N * np.sum(x, axis=0)
        xmu = x - mu

        sq = xmu ** 2
        var = 1. / N * np.sum(sq, axis=0)
        std = np.sqrt(var + eps)
        ivar = 1. / std
        xhat = xmu * ivar

        gmx = gm * xhat
        out = gammax + bt

        return out, xhat, xmu, ivar, std, var, eps

    def forward_test(x, gm, bt, eps, rm, rv):
        nx = x - rm / (np.sqrt(rv + eps))
        return gm * nx + bt

    def backward(dout, gm, xhat, xmu, ivar, std, var, eps):
        N = dout.shape[0]

        dbeta = np.sum(dout, axis=0)

        dgamma = np.sum(dout * xhat, axis=0)

        dxhat = dgammax * gm
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (std ** 2) * divar
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
        dsq = 1. / N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1. / N * np.ones(dout.shape) * dmu
        dx = dx1 + dx2

        return dx, dgamma, dbeta

#%% OPTIMIZATION
def vanilla(w, dw, lr=1):
    return w - lr * dw

def momentum(w, dw, ldw, lr=1, m=0):
    nw = w - dw * lr + ldw * m
    return nw, dw

def rmsProp(w, dx, lr, dr, eps, lgt=0):
    gt = dr * lgt + (1 - dr) * dx**2
    nw = w - lr * dx / (np.sqrt(gt) + eps)
    return nw, gt

def adam(w, dx, lr, bt1, bt2, eps, lm=0, lv=0):
    m = bt1 * lm + (1 - b1) * dx
    v = bt2 * lv + (1 - b2) * (dx ** 2)

    nw = w - lr * m / (np.sqrt(v) + eps)
    return nw, m, v

#%% REGULARIZATION
def dropout(x, probs, isTesting=False):
    if isTesting:
        return x * probs

    dp = np.random.random(x.shape) - probs
    return (dp < 0) * x
