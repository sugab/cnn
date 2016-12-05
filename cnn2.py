# Copyright (c) Bagus Cahyono 2016

# Importing Libs
import numpy as np
from scipy import signal as sg
from im2col import *

#%% CONFIG CLASS
# class implementation for saving configuration
class CONFIG:
    # learning rate
    lr = 1
    momentum = 0
    # dropout probability
    p = 0

# MAIN CNN CLASS
class CNN:

    # Initialize CNN with given parameters
    # lr for learning rate
    # momentum for momentum value
    def __init__(self, model, scoring, data, lr=1, momentum=0, dropout=0):
        self.model = model
        self.scoring = scoring

        if len(data) == 2:
            self.x, self.labels = data
            self.v_data, self.v_lables = (None, None)
        elif len(data) == 4:
            self.x, self.labels, self.v_data, self.v_labels = data

        # Saving learning rate & momentum for CONFIG
        CONFIG.lr = lr
        CONFIG.momentum = momentum
        CONFIG.p = dropout

    # Run learning using SGD
    def run(self, epoch, batch_size=1, debug=True):
        # Get the depth of the model
        model_depth = len(self.model)

        # Init array for saving result
        t_result = np.zeros((epoch, len(self.x) / batch_size))
        v_result = np.zeros((epoch, 2))

        # run each epoch
        for i in xrange(epoch):
            # un each batch
            for j in xrange(len(self.x) / batch_size):
                # Get single data
                # Get data based on batch size
                x_batch = self.x[batch_size*j:batch_size*(j+1)]
                label = np.array([self.labels[batch_size*j:batch_size*(j+1)]])

                # Run forward propagation
                y = x_batch
                for m in xrange(model_depth):
                    # forward each layer
                    y = self.model[m].forward(y)

                    # Check if using dropout
                    if CONFIG.p != 0 and m < model_depth - 1:
                        u = (np.random.rand(*y.shape) < CONFIG.p) / CONFIG.p
                        y *= u

                # Calculate score and loss with given scoring class
                score, loss = self.scoring.forward(y, label)
                # Save loss to array
                t_result[i, j] = loss

                # print loss if debug is true
                if debug:
                    print i, loss

                # Calculate error with given scoring class
                d = self.scoring.backward(score, label)

                # Run backward propagation
                for m in xrange(model_depth):
                    # Backprop each layer
                    d = self.model[model_depth-m-1].backward(d)

            #Validation for each epoch
            if self.v_data is not None and self.v_labels is not None:
                # Run forward propagation for Validation
                y = self.v_data
                for m in xrange(model_depth):
                    y = self.model[m].forward(y)

                # Calculate score and loss
                score, loss = self.scoring.forward(y, self.v_labels)
                # Calculate accuracy
                acc = np.sum((np.argmax(score, axis=1) == self.v_labels) * 1) / float(len(self.v_labels))
                # save loss and accuracy in array
                v_result[i, 0] = loss
                v_result[i, 1] = acc

                # Print the result
                print "EPOCH %s" % i
                print "V_LOSS = %s" % loss
                print "V_ACCURACY = %s" % acc

        return t_result, v_result

#%% HELPER FUNCTION
# Flatten 4D matrix to 2D matrix
def flaten(x):
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

# Unflatter 2D matrix to given shape (4D)
def unflaten(x, shape):
    return x.reshape(shape)

# Zero Pad
# Adding zero padding to 4D matrix
def zero_pad(x, p):
    # Input matrix x must 4D
    assert x.ndim == 4, 'Input not 4D matrix'

    # Define which part of the matrix we want to pad
    npad = ((0,0), (p,p), (p,p), (0,0))
    # Return padding calculation result
    return np.pad(x, pad_width=npad, mode='constant', constant_values=0)

# Size After Convolve
# Calculate size of given x after forward calculation with given f=filter, p=padding, s=stride
def size_after_forward(x, f, p=0, s=1):
    return (x - f + 2 * p) / s + 1

# Size Before Convolve
# Calculate size of given x after backward calculation with given f=filter, p=padding, s=stride
def size_after_backward(x, f, p=0, s=1):
    return x * s + f - s - 2 * p

# Max Pooling
# Calculate Max Pooling of given x=input (3D Matrix) with given f=filter size and s=stride
def max_pooling(x, h, w):
    # Input matrix x must 3D
    assert x.ndim == 3, 'Input not 3D matrix'

    # Calculate result dimension
    depth = x.shape[0]
    height = size_after_forward(x.shape[1], h, s=h)
    width = size_after_forward(x.shape[2], w, s=w)
    # Initialize result & switch with zero value
    result = np.zeros((depth, height, width))
    switch = np.zeros((depth, height, width))

    # Loop through depth of the input
    for i in xrange(depth):
        # Loop through height of the input
        for j in xrange(height):
            # Loop through width of the input
            for k in xrange(width):
                # Calculate max value from filter area of the input
                result[i, j, k] = np.max(x[i, j*h:j*h+h, k*w:k*w+w])
                # Keep track of max index
                switch[i, j, k] = np.argmax(x[i, j*h:j*h+h, k*w:k*w+w])

    # Return result and switch value
    return (result, switch)

# Max Pooling
# Calculate Max Pooling of given x=input (3D Matrix) with given f=filter size and s=stride
def unmax_pooling(x, switch, h, w):
    # Input matrix x must 3D
    assert x.ndim == 3, 'Input not 3D matrix'

    # Calculate result dimension
    depth = x.shape[0]
    height = size_after_backward(x.shape[1], h, s=h)
    width = size_after_backward(x.shape[2], w, s=w)
    # Initialize result with zero value
    result = np.zeros((depth, height, width))

    # Loop through depth of the input
    for i in xrange(x.shape[0]):
        # Loop through height of the input
        for j in xrange(x.shape[1]):
            # Loop through width of the input
            for k in xrange(x.shape[2]):
                # Calculate max value from filter area of the input
                r_index = int((j * h) + (switch[i, j, k] / w))
                c_index = int((k * w) + (switch[i, j, k] % w))

                result[i, r_index, c_index] = x[i, j, k]

    # Return result and switch value
    return result

#%% LAYER CLASS
# Class implementation for FullyConnectedLayer
class FCLayer:

    # Intialize FCLayer with given parameters
    # w for initial weight
    # b for initial bias
    # activation for activation function like ReLU, Sigmoid, etc. Default = None
    def __init__(self, w, b, activation=None):
        self.w = w
        self.b = b
        self.activation = activation
        # Initialize last dw = 0 for momentum
        self.ldw = 0

    # Forward FC
    # Calculate forward FC for given x
    def forward(self, x):
        self.x = x

        # Check if x is still 4D matrix
        if self.x.ndim == 4:
            # if true then flaten to 2D matrix
            self.x = flaten(self.x)

        net = np.dot(self.x, self.w) + self.b

        # Check wether this layer use activation or not
        if self.activation is None:
            self.y = net
        else:
            self.y = self.activation.forward(net)

        return self.y

    # Backward FC
    # Calculate backward FC for given error
    # Also currenly auto updating the w
    def backward(self, e):
        self.d = e

        # Check wether this layer use activation or not
        if self.activation is not None:
            # If use any activation layer
            # then backprop through it
            self.d *= self.activation.backward(self.y)

        # Calculate dx
        dx = self.d.dot(self.w.T)

        # Calculate gradient using momentum
        self.dw = self.x.T.dot(self.d) * CONFIG.lr + self.ldw * CONFIG.momentum
        # Updating weight and bias
        self.w -= self.dw
        self.b -= np.sum(self.d, axis=0) * CONFIG.lr
        # saving last dw for later
        self.ldw = self.dw

        # Return the error for previous layer
        return dx

# Class implementation for ConvolutionLayer
class ConvolutionLayer:

    # Initialize ConvolutionLayer with given parameters
    # w for initial weight
    # b for initial bias
    # p for padding default = 0 (No Padding)
    def __init__(self, W, b, p=0, s=1, activation=None):
        self.W = W
        self.b = b
        self.p = p
        self.s = s
        self.activation = activation
        # Initialize last dw = 0 for momentum
        self.ldW = 0

    # Forward Convolution
    # Calculate feed forward convolution layer for given x=input, filters, and p=padding
    def forward(self, X):
        stride = 1
        self.X = X

        n_filters, d_filter, h_filter, w_filter = self.W.shape
        n_x, d_x, h_x, w_x = self.X.shape
        h_out = (h_x - h_filter + 2 * self.p) / self.s + 1
        w_out = (w_x - w_filter + 2 * self.p) / self.s + 1

        # if not isinstance(h_out, int) or not isinstance(w_out, int):
        #     raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        self.X_col = im2col_indices(self.X, h_filter, w_filter, padding=self.p, stride=self.s)
        W_col = self.W.reshape(n_filters, -1)

        self.out = W_col.dot(self.X_col) + self.b
        self.out = self.out.reshape(n_filters, h_out, w_out, n_x)
        self.out = self.out.transpose(3, 0, 1, 2)

        # Call ReLU function (Activation function) for each value
        # Return the ReLU result
        return self.activation.forward(self.out)

    # Backward Convolution
    # Calculate gradient and delta from given parameters
    def backward(self, dout):
        n_filter, d_filter, h_filter, w_filter = self.W.shape

        dout *= self.activation.backward(self.out)

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped.dot(self.X_col.T)
        dW = dW.reshape(self.W.shape)

        W_reshape = self.W.reshape(n_filter, -1)
        dX_col = W_reshape.T.dot(dout_reshaped)
        dX = col2im_indices(dX_col, self.X.shape, h_filter, w_filter, padding=self.p, stride=self.s)

        # Adding momentum to gradient
        dW = dW * CONFIG.lr + self.ldW * CONFIG.momentum
        # Updating gradient
        self.W -= dW
        self.b -= db * CONFIG.lr
        # saving last dw for later
        self.ldW = dW

        # Return the delta result
        return dX

# Class implementaion for PoolingLayer
class PoolingLayer:

    # Initialize PollingLayer with given parameters
    # h for filter height
    # w for filter weight
    def __init__(self, h, w):
        self.h = h
        self.w = w

    # Forward Pooling
    # Calculate feed forward pooling layer
    def forward(self, x):
        # Initialize result dimension after pooling
        self.y = np.zeros((x.shape[0], x.shape[1], size_after_forward(x.shape[2], self.h, s=self.h), size_after_forward(x.shape[3], self.w, s=self.w)))
        self.sw = np.zeros(self.y.shape)

        # loop through all input and do max pooling
        for i in xrange(x.shape[0]):
            # Do max pooling
            self.y[i], self.sw[i] = max_pooling(x[i], self.h, self.w)

        # return the result
        return self.y

    # Backward Pooling
    # Calculate backward pooling layer
    def backward(self, e):
        # Check if e is still 2D matrix
        if e.ndim == 2:
            # if true then reshape to 4D array
            e = unflaten(e, self.y.shape)

        # Initialize result dimension after backward pool
        y = np.zeros((e.shape[0], e.shape[1], size_after_backward(e.shape[2], self.h, s=self.h), size_after_backward(e.shape[3], self.w, s=self.w)))

        # loop through all input and do unmax pooling
        for i in xrange(e.shape[0]):
            # Do unmax pooling
            y[i] = unmax_pooling(e[i], self.sw[i], self.h, self.w)

        # return the result
        return y

#%% SCORING LAYER
# Class implementaion for Softmax
class SoftmaxLayer:
    # Forward Softmax
    # Calculate forward Softmax Layer
    @staticmethod
    def forward(x, target):
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Calculate data loss
        corect_logprobs = -np.log(probs[range(len(target)), target])
        data_loss = np.sum(corect_logprobs)
        data_loss *= 1. / len(target)

        # Return the probability and data loss
        return probs, data_loss

    # Backward Softmax
    # Calculate backward Softmax Layer
    @staticmethod
    def backward(x, target):
        d = x
        d[range(len(target)), target] -= 1

        # Return the scores after backward
        return d

#%% ACTIVATION LAYER
# Class implementation for ReLU layer
class ReLULayer:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0) * 1

class ELULayer:
    @staticmethod
    def forward(x):
        return np.where(x >= 0, x, 0.1 * (np.exp(x) - 1))

    @staticmethod
    def backward(x):
        return np.where(x >= 0, 1, x + 0.1)

class LeakyReLULayer:
    @staticmethod
    def forward(x):
        return np.maximum(x * 0.01, x)

    @staticmethod
    def backward(x):
        return np.where(x >= 0, 1, 0.01)
