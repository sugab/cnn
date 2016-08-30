# Copyright (c) Bagus Cahyono 2016

# Importing Library
import numpy as np
from scipy import signal as sg

# MARK - FUNCTION DEFINITION
# Sigmoid
def sigmoid(x,deriv=False):
    # if the deriv==True then calculate the derivative instead
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# Rectified Linear Unit
# Calculate relu of given x
def relu(x, deriv=False):
    # if the deriv==True then calculate the derivative instead
    if (deriv):
        return (x > 0) * 1

    return np.maximum(0, x)

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
def max_pooling(x, f, s=2):
    # Input matrix x must 3D
    assert x.ndim == 3, 'Input not 3D matrix'

    # Calculate result dimension
    depth = x.shape[2]
    height = size_after_forward(x.shape[0], f, s=s)
    width = size_after_forward(x.shape[1], f, s=s)
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
                result[j, k, i] = np.max(x[j*s:j*s+f, k*s:k*s+f, i])
                # Keep track of max index
                switch[j, k, i] = np.argmax(x[j*s:j*s+f, k*s:k*s+f, i])

    # Return result and switch value
    return (result, switch)

# Max Pooling
# Calculate Max Pooling of given x=input (3D Matrix) with given f=filter size and s=stride
def unmax_pooling(x, switch, f, s=2):
    # Input matrix x must 3D
    assert x.ndim == 3, 'Input not 3D matrix'

    # Calculate result dimension
    depth = x.shape[2]
    height = size_after_backward(x.shape[0], f, s=s)
    width = size_after_backward(x.shape[1], f, s=s)
    # Initialize result with zero value
    result = np.zeros((height, width, depth))

    # Loop through depth of the input
    for i in xrange(x.shape[2]):
        # Loop through height of the input
        for j in xrange(x.shape[0]):
            # Loop through width of the input
            for k in xrange(x.shape[1]):
                # Calculate max value from filter area of the input
                r_index = (j * s) + (switch[j, k, i] / f)
                c_index = (k * s) + (switch[j, k, i] % f)

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
            delta_result[i] += r if p <= 0 else r[p:-p, p:-p, :]

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
    for i in xrange(x.shape[3]):
        # sum delta_y
        bias_update_result[i] = np.sum(delta_y[:, :, :, i])

    # Return the delta and gradient result
    return (delta_result, gradient_result, bias_update_result)


# Forward Pooling
# Calculate feed forward pooling layer
def forward_pool(x, filter_size, stride):
    # Initialize result dimension after pooling
    result = np.zeros((x.shape[0], size_after_forward(x.shape[1], filter_size, s=stride), size_after_forward(x.shape[2], filter_size, s=stride), x.shape[3]))
    switch = np.zeros(result.shape)

    # loop through all input and do max pooling
    for i in xrange(x.shape[0]):
        # Do max pooling
        result[i], switch[i] = max_pooling(x[i], filter_size, stride)

    # return the result
    return (result, switch)

# Backward Pooling
# Calculate backward pooling layer
def backward_pool(x, switch, filter_size, stride):
    # Initialize result dimension after backward pool
    result = np.zeros((x.shape[0], size_after_backward(x.shape[1], filter_size, s=stride), size_after_backward(x.shape[2], filter_size, s=stride), x.shape[3]))

    # loop through all input and do unmax pooling
    for i in xrange(x.shape[0]):
        # Do unmax pooling
        result[i] = unmax_pooling(x[i], switch[i], filter_size, stride)

    # return the result
    return result
