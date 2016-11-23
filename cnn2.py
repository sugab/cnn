# Copyright (c) Bagus Cahyono 2016

# Importing Libs
import numpy as np
from scipy import signal as sg

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
    def runSGD(self, epoch, batch_size=1, debug=True):
        # Get the depth of the model
        model_depth = len(self.model)

        # Init array for saving result
        t_result = np.zeros((epoch, len(self.x)))
        v_result = np.zeros((epoch, 2))

        # run each epoch
        for i in xrange(epoch):
            # un each batch
            for j in xrange(len(self.x) / batch_size):
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

                print y

                # Calculate score and loss with given scoring class
                score, loss = self.scoring.forward(y, label)
                # Save loss to array
                t_result[i, j] = loss

                # print loss if debug is true
                if debug:
                    print "EPOCH %s TARGET : %s PREDICTION : %s LOSS = %s" % (i, label, np.argmax(score), loss)

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
                print "V_RESULT = %s" % score
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
    depth = x.shape[2]
    height = size_after_forward(x.shape[0], h, s=h)
    width = size_after_forward(x.shape[1], w, s=w)
    # Initialize result & switch with zero value
    result = np.zeros((height, width, depth))
    switch = np.zeros((height, width, depth))

    # Loop through depth of the input
    for i in xrange(depth):
        # Loop through height of the input
        for j in xrange(height):
            # Loop through width of the input
            for k in xrange(width):
                # Calculate max value from filter area of the input
                result[j, k, i] = np.max(x[j*h:j*h+h, k*w:k*w+w, i])
                # Keep track of max index
                switch[j, k, i] = np.argmax(x[j*h:j*h+h, k*w:k*w+w, i])

    # Return result and switch value
    return (result, switch)

# Max Pooling
# Calculate Max Pooling of given x=input (3D Matrix) with given f=filter size and s=stride
def unmax_pooling(x, switch, h, w):
    # Input matrix x must 3D
    assert x.ndim == 3, 'Input not 3D matrix'

    # Calculate result dimension
    depth = x.shape[2]
    height = size_after_backward(x.shape[0], h, s=h)
    width = size_after_backward(x.shape[1], w, s=w)
    # Initialize result with zero value
    result = np.zeros((height, width, depth))

    # Loop through depth of the input
    for i in xrange(x.shape[2]):
        # Loop through height of the input
        for j in xrange(x.shape[0]):
            # Loop through width of the input
            for k in xrange(x.shape[1]):
                # Calculate max value from filter area of the input
                r_index = int((j * h) + (switch[j, k, i] / w))
                c_index = int((k * w) + (switch[j, k, i] % w))

                result[r_index, c_index, i] = x[j, k, i]

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
    def __init__(self, w, b, p=0):
        self.w = w
        self.b = b
        self.p = p
        # Initialize last dw = 0 for momentum
        self.ldw = 0

    # Forward Convolution
    # Calculate feed forward convolution layer for given x=input, filters, and p=padding
    def forward(self, x):
        # Input matrix x must 4D
        assert x.ndim == 4, 'Input not 4D matrix'

        self.x = x

        # Initialize result value with size according to x and filters
        self.y = np.zeros((self.x.shape[0], size_after_forward(self.x.shape[1], self.w.shape[1], self.p), size_after_forward(self.x.shape[2], self.w.shape[2], self.p), self.w.shape[0]))

        # Adding zero padding if p != 0
        if (self.p > 0): self.x = zero_pad(self.x, self.p)

        # Convolve each input with each filter
        # Looping through each input
        for i in xrange(self.x.shape[0]):
            # Looping through each filters
            for j in xrange(self.w.shape[0]):
                # Do convolve computation
                # 'valid' means that there is no zero padding
                # Convolve computation resulting 3D matrix so convert it to 2D matrix
                self.y[i, :, :, j] = (sg.convolve(self.x[i], self.w[j], 'valid')).reshape((self.y.shape[1], self.y.shape[2])) + self.b[j]

        # Call ReLU function (Activation function) for each value
        # Return the ReLU result
        return ReLULayer.forward(self.y)

    # Backward Convolution
    # Calculate gradient and delta from given parameters
    def backward(self, e):

        # Check if e is still 2D matrix
        if e.ndim == 2:
            # if true then reshape to 4D array
            e = unflaten(e, self.y.shape)

        # Backprop through relu layer first
        self.dy = ReLULayer.backward(self.y) * e

        # Initialize result variable
        self.delta = np.zeros(self.x.shape)
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros((self.dy.shape[3]))

        # Loop through each delta_y
        for i in xrange(self.dy.shape[0]):
            # Loop through each filters
            # Filters count == dconv
            for j in xrange(self.w.shape[0]):
                # Get single layer(2D) from delta input
                s = self.dy[i, :, :, j].reshape((self.dy.shape[1], self.dy.shape[2], 1))
                # Flip the filters along its axis
                f = np.fliplr(np.flipud(self.w[j]))
                # Full convolve the delta layer with its corresponded filter
                r = sg.convolve(s, f, 'full')
                # Ignore pad if there is a padding in forward conv
                self.delta[i] += r

        # Loop through each image input
        # image count == delta count
        for i in xrange(self.x.shape[0]):
            # Loop through each delta depth
            for j in xrange(self.dy.shape[3]):
                # Get single layer(2D) from delta input
                f = self.dy[i, :, :, j].reshape((self.dy.shape[1], self.dy.shape[2], 1))
                # Convolve image input with its corresponded delta from next layer
                self.dw[j] += sg.convolve(self.x[i], f, 'valid')

        # Calculate bias update
        for i in xrange(self.w.shape[0]):
            # sum delta_y
            self.db[i] = np.sum(self.dy[:, :, :, i])

        # Trim delta result if padded
        if (self.p > 0): self.delta = self.delta[:, self.p:-self.p, self.p:-self.p, :]

        # Adding momentum to gradient
        self.dw = self.dw * CONFIG.lr + self.ldw * CONFIG.momentum
        # Updating gradient
        self.w -= self.dw
        self.b -= self.db * CONFIG.lr
        # saving last dw for later
        self.ldw = self.dw

        # Return the delta result
        return self.delta

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
        self.y = np.zeros((x.shape[0], size_after_forward(x.shape[1], self.h, s=self.h), size_after_forward(x.shape[2], self.w, s=self.w), x.shape[3]))
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
        y = np.zeros((e.shape[0], size_after_backward(e.shape[1], self.h, s=self.h), size_after_backward(e.shape[2], self.w, s=self.w), e.shape[3]))

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
