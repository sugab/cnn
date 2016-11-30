# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:07:02 2016

@author: baguscahyono
"""

import numpy as np
import cnn
import mnist

tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 200), digits=[0,1])
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(200, 250), digits=[0,1])

#%% INIIT FILTER

np.random.seed(1)

l0 = tr_data[:, :, :, np.newaxis]
lb = tr_label[:, np.newaxis]

filter_1 = np.random.normal(0, 0.1, (10, 5, 5, 1))
bias_1 = np.ones((10))
filter_3 = np.random.normal(0, 0.1, (10, 3, 3, 10))
bias_3 = np.ones((10))
filter_4 = np.random.normal(0, 0.1, (10, 3, 3, 10))
bias_4 = np.ones((10))
filter_6 = np.random.normal(0, 0.1, (10, 3, 3, 10))
bias_6 = np.ones((10))
syn7 = np.random.normal(0, 0.1, (40, 20))
bias_7 = np.ones((1, 20))
syn8 = np.random.normal(0, 0.1, (20, 1))
bias_8 = np.ones((1, 1))

lsyn8 = 0
lsyn7 = 0
lg6 = 0
lg4 = 0
lg3 = 0
lg1 = 0

#%% RUN EPOCH

lr = 0.005
momentum = 0.5

for i in xrange(50):

    l1 = cnn.forward_conv(l0, filter_1, bias_1)
    l2, l2_switches = cnn.forward_pool(l1, 2, 2)
    l3 = cnn.forward_conv(l2, filter_3, bias_3)
    l4 = cnn.forward_conv(l3, filter_4, bias_4)
    l5, l5_switches = cnn.forward_pool(l4, 2, 2)
    l6 = cnn.forward_conv(l5, filter_6, bias_6)

    l7_in = l6.reshape((l6.shape[0], l6.shape[1] * l6.shape[2] * l6.shape[3]))

    l7 = cnn.relu(np.dot(l7_in, syn7) + bias_7)
    l8_na = np.dot(l7, syn8) + bias_8
    l8 = cnn.sigmoid(l8_na)

    e = l8 - lb
    
#    print "%s" % (i,)
#    print "%s -> %s" % (l8_na, l8)

    ld8 = e * cnn.sigmoid(l8, deriv=True)
    ld7 = np.dot(ld8, syn8.T) * cnn.relu(l7, deriv=True)
    ld6_out = np.dot(ld7, syn7.T)

    ld6_r = ld6_out.reshape((l6.shape[0], l6.shape[1], l6.shape[2], l6.shape[3]))

    ld6, g6, b6 = cnn.backward_conv(l5, l6, ld6_r, filter_6)
    ld5 = cnn.backward_pool(ld6, l5_switches, 2, 2)
    ld4, g4, b4 = cnn.backward_conv(l3, l4, ld5, filter_4)
    ld3, g3, b3 = cnn.backward_conv(l2, l3, ld4, filter_3)
    ld2 = cnn.backward_pool(ld3, l2_switches, 2, 2)
    ld1, g1, b1 = cnn.backward_conv(l0, l1, ld2, filter_1)

    dsyn8 = l7.T.dot(ld8) + lsyn8 * momentum
    dsyn7 = l7_in.T.dot(ld7) + lsyn7 * momentum

    syn8 -= dsyn8 * lr
    syn7 -= dsyn7 * lr
    bias_8 -= np.sum(ld8, axis=0) * lr
    bias_7 -= np.sum(ld7, axis=0) * lr
    
    lsyn8 = dsyn8
    lsyn7 = dsyn7
    
    g6 += lg6 * momentum
    g4 += lg4 * momentum
    g3 += lg3 * momentum
    g1 += lg1 * momentum

    filter_6 -= g6 * lr
    filter_4 -= g4 * lr
    filter_3 -= g3 * lr
    filter_1 -= g1 * lr    
    bias_6 -= b6 * lr
    bias_4 -= b4 * lr
    bias_3 -= b3 * lr
    bias_1 -= b1 * lr
    
    lg6 = g6
    lg4 = g4
    lg3 = g3
    lg1 = g1

print l0.shape
print l1.shape
print l2.shape
print l3.shape
print l4.shape
print l5.shape
print l6.shape
print l7.shape
print l8.shape
print "-"

cf = np.zeros((2, 2))
for i in xrange(lb.shape[0]):
    cf[int(round(lb[i])), int(round(l8[i]))] += 1

print "Accuracy Training: %f" % ((cf[0,0] + cf[1,1]) / float(lb.shape[0]))