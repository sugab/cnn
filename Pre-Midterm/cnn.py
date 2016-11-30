# Copyright (c) Bagus Cahyono 2016

# Importing Library
import numpy as np
from scipy import signal as sg

# MARK - FUNCTION DEFINITION
# Sigmoid
def sigmoid(x, deriv=False):
    # if the deriv==True then calculate the derivative instead
    if(deriv):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def tanh(x, deriv=False):
    if (deriv):
        return 1 - (x ** 2)

    return 2 / (1 + np.exp(-2 * x)) - 1

# Rectified Linear Unit
# Calculate relu of given x
def relu(x, deriv=False):
    # if the deriv==True then calculate the derivative instead
    if (deriv):
        return (x > 0) * 1

    return np.maximum(0, x)

def elu(x, a=0.1, deriv=False):
    if (deriv):
        return np.where(x >= 0, 1, x + a)

    return np.where(x >= 0, x, a * (np.exp(x) - 1))

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

# Forward Convolution
# Calculate feed forward convolution layer for given x=input, filters, and p=padding
def forward_conv(x, filters, bias, p=0):
    # Initialize result value with size according to x and filters
    result = np.zeros((x.shape[0], size_after_forward(x.shape[1], filters.shape[1], p), size_after_forward(x.shape[2], filters.shape[2], p), filters.shape[0]))

    # Adding zero padding if p != 0
    if (p > 0): x = zero_pad(x, p)

    # Convolve each input with each filter
    # Looping through each input
    for i in xrange(x.shape[0]):
        # Looping through each filters
        for j in xrange(filters.shape[0]):
            # Do convolve computation
            # 'valid' means that there is no zero padding
            # Convolve computation resulting 3D matrix so convert it to 2D matrix
            result[i, :, :, j] = (sg.convolve(x[i], filters[j], 'valid')).reshape((result.shape[1], result.shape[2])) + bias[j]

    # Call ReLU function (Activation function) for each value
    # Return the ReLU result
    return relu(result)

# Backward Convolution
# Calculate gradient and delta from given parameters
def backward_conv(x, y, error_y, filters, p=0, s=1):
    # Backprop through relu layer first
    delta_y = relu(y, deriv=True) * error_y

    # Adding zero padding if p != 0
    if (p > 0): x = zero_pad(x, p)

    # Initialize result variable
    delta_result = np.zeros(x.shape)
    gradient_result = np.zeros(filters.shape)
    bias_update_result = np.zeros((delta_y.shape[3]))

    # Loop through each delta_y
    for i in xrange(delta_y.shape[0]):
        # Loop through each filters
        # Filters count == dconv
        for j in xrange(filters.shape[0]):
            # Get single layer(2D) from delta input
            s = delta_y[i, :, :, j].reshape((delta_y.shape[1], delta_y.shape[2], 1))
            # Flip the filters along its axis
            f = np.fliplr(np.flipud(filters[j]))
            # Full convolve the delta layer with its corresponded filter
            r = sg.convolve(s, f, 'full')
            # Ignore pad if there is a padding in forward conv
            delta_result[i] += r

    # Loop through each image input
    # image count == delta count
    for i in xrange(x.shape[0]):
        # Loop through each delta depth
        for j in xrange(delta_y.shape[3]):
            # Get single layer(2D) from delta input
            f = delta_y[i, :, :, j].reshape((delta_y.shape[1], delta_y.shape[2], 1))
            # Convolve image input with its corresponded delta from next layer
            gradient_result[j] += sg.convolve(x[i], f, 'valid')

    # Calculate bias update
    for i in xrange(filters.shape[0]):
        # sum delta_y
        bias_update_result[i] = np.sum(delta_y[:, :, :, i])

    # Trim delta result if padded
    if (p > 0): delta_result = delta_result[:, p:-p, p:-p, :]

    # Return the delta and gradient result
    return (delta_result, gradient_result, bias_update_result)


# Forward Pooling
# Calculate feed forward pooling layer
def forward_pool(x, h, w):
    # Initialize result dimension after pooling
    result = np.zeros((x.shape[0], size_after_forward(x.shape[1], h, s=h), size_after_forward(x.shape[2], w, s=w), x.shape[3]))
    switch = np.zeros(result.shape)

    # loop through all input and do max pooling
    for i in xrange(x.shape[0]):
        # Do max pooling
        result[i], switch[i] = max_pooling(x[i], h, w)

    # return the result
    return (result, switch)

# Backward Pooling
# Calculate backward pooling layer
def backward_pool(x, switch, h, w):
    # Initialize result dimension after backward pool
    result = np.zeros((x.shape[0], size_after_backward(x.shape[1], h, s=h), size_after_backward(x.shape[2], w, s=w), x.shape[3]))

    # loop through all input and do unmax pooling
    for i in xrange(x.shape[0]):
        # Do unmax pooling
        result[i] = unmax_pooling(x[i], switch[i], h, w)

    # return the result
    return result

# Forward Softmax
# Calculate forward Softmax Layer
def forward_softmax(x, target):
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
def backward_softmax(x, target):
    d = x
    d[range(len(target)), target] -= 1

    # Return the scores after backward
    return d
