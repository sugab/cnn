# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:52:58 2016

@author: baguscahyono
"""

import numpy as np
import mnist
import cnn

tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 5))
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(1000, 1100))

#%% INIT FILTER

np.random.seed(1)

l0 = tr_data.reshape((tr_data.shape[0], 784))
lb = tr_label

syn1 = np.random.normal(0, 0.01, (784, 196))
bias1 = np.zeros(196)
syn2 = np.random.normal(0, 0.01, (196, 49))
bias2 = np.zeros(49)
syn3 = np.random.normal(0, 0.01, (49, 10))
bias3 = np.zeros(10)

lsyn1 = 0
lsyn2 = 0
lsyn3 = 0

#%% TRAINING

lr = 0.005
momentum = 0

for i in xrange(1):

    l1 = cnn.sigmoid(np.dot(l0, syn1) + bias1)
    l2 = cnn.sigmoid(np.dot(l1, syn2) + bias2)
    l3_n = np.dot(l2, syn3) + bias3

    exp_scores = np.exp(l3_n)
    l3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    result = np.argmax(l3, axis=1)

    print result

    ld3 = l3
    ld3[range(len(lb)), lb] -= 1

    print ld3

    ld2 = np.dot(ld3, syn3.T) * cnn.sigmoid(l2, deriv=True)
    ld1 = np.dot(ld2, syn2.T) * cnn.sigmoid(l1, deriv=True)

    dsyn3 = l2.T.dot(ld3) + lsyn3 * momentum
    dsyn2 = l1.T.dot(ld2) + lsyn2 * momentum
    dsyn1 = l0.T.dot(ld1) + lsyn1 * momentum

    syn3 -= dsyn3 * lr
    syn2 -= dsyn2 * lr
    syn1 -= dsyn1 * lr

    bias3 -= np.sum(ld3, axis=0) * lr
    bias2 -= np.sum(ld2, axis=0) * lr
    bias1 -= np.sum(ld1, axis=0) * lr

    lsyn3 = dsyn3
    lsyn2 = dsyn2
    lsyn1 = dsyn1


#%% CALCULATE ACCURACY
cft = np.zeros((10, 10))
for j in xrange(len(lb)):
    cft[result[j], lb[j]] += 1

print "Training Data EPOCH: %s = %s" % (i + 1, np.sum(np.diag(cft)))

l0 = vl_data.reshape((vl_data.shape[0], 784))
lb = vl_label[:, np.newaxis]

l1 = cnn.sigmoid(np.dot(l0, syn1) + bias1)
l2 = cnn.sigmoid(np.dot(l1, syn2) + bias2)
l3 = cnn.sigmoid(np.dot(l2, syn3) + bias3)

result = np.argmax(l3, axis=1)
cfv = np.zeros((10, 10))
for i in xrange(len(lb)):
    cfv[result[i], lb[i]] += 1

print "Validation Accuracy: %s" % np.sum(np.diag(cfv))
