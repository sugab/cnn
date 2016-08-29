# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 12:52:58 2016

@author: baguscahyono
"""

import numpy as np
import mnist
import cnn

tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 100), digits=[0,1])
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(0, 100), digits=[0,1])

#%% INIIT FILTER

np.random.seed(1)

l0 = tr_data.reshape((tr_data.shape[0], 784))
lb = tr_label[:, np.newaxis]

syn0 = np.random.normal(0, 0.01, (784, 196))
bias_0 = np.zeros((1, 196))
syn1 = np.random.normal(0, 0.01, (196, 49))
bias_1 = np.zeros((1, 49))
syn2 = np.random.normal(0, 0.01, (49, 1))
bias_2 = np.zeros((1, 1))

lsyn0 = 0
lsyn1 = 0
lsyn2 = 0

#%% TRAINING

lr = 0.01
momentum = 0.9

for i in xrange(60):
    
    l1 = cnn.sigmoid(np.dot(l0, syn0) + bias_0)
    l2 = cnn.sigmoid(np.dot(l1, syn1) + bias_1)
    l3 = cnn.sigmoid(np.dot(l2, syn2) + bias_2)

    e = l3 - lb
    
#    print "%s" % (i,)
#    print "%s" % (l3)

    ld3 = e * cnn.sigmoid(l3, deriv=True)
    ld2 = np.dot(ld3, syn2.T) * cnn.relu(l2, deriv=True)
    ld1 = np.dot(ld2, syn1.T) * cnn.relu(l1, deriv=True)
    
    dsyn2 = l2.T.dot(ld3) + lsyn2 * momentum
    dsyn1 = l1.T.dot(ld2) + lsyn1 * momentum
    dsyn0 = l0.T.dot(ld1) + lsyn0 * momentum

    syn2 -= dsyn2 * lr
    syn1 -= dsyn1 * lr
    syn0 -= dsyn0 * lr
    
    bias_2 -= np.sum(ld3, axis=0) * lr
    bias_1 -= np.sum(ld2, axis=0) * lr
    bias_0 -= np.sum(ld1, axis=0) * lr
    
    lsyn2 = dsyn2
    lsyn1 = dsyn1
    lsyn0 = dsyn0


#%% CALCULATE ACCURACY
    
cf = np.zeros((2, 2))
for i in xrange(lb.shape[0]):
    cf[int(round(lb[i])), int(round(l3[i]))] += 1

print "Accuracy Training: %f" % ((cf[0,0] + cf[1,1]) / float(lb.shape[0]))

l0 = vl_data.reshape((vl_data.shape[0], 784))
lb = vl_label[:, np.newaxis]

l1 = cnn.sigmoid(np.dot(l0, syn0) + bias_0)
l2 = cnn.sigmoid(np.dot(l1, syn1) + bias_1)
l3 = cnn.sigmoid(np.dot(l2, syn2) + bias_2)

cf = np.zeros((2, 2))
for i in xrange(lb.shape[0]):
    cf[int(round(lb[i])), int(round(l3[i]))] += 1

print "Accuracy Validation: %f" % ((cf[0,0] + cf[1,1]) / float(lb.shape[0]))
