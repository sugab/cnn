import numpy as np
import helper as h
import cnn2 as c
import mnist

tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 100))
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(100, 110))

tr_data = tr_data[:, :, :, np.newaxis]
vl_data = vl_data[:, :, :, np.newaxis]

np.random.seed(2)

l1 = c.ConvolutionLayer(h.random((16, 5, 5, 1)), np.ones((16)))
l2 = c.PoolingLayer(2, 2)
l3 = c.ConvolutionLayer(h.random((32, 3, 3, 16)), np.ones((32)))
l4 = c.ConvolutionLayer(h.random((32, 3, 3, 32)), np.ones((32)))
l5 = c.PoolingLayer(2, 2)
l6 = c.ConvolutionLayer(h.random((32, 3, 3, 32)), np.ones((32)))
l7 = c.FCLayer(h.random((128, 128)), np.ones(128), activation=c.ReLULayer)
l8 = c.FCLayer(h.random((128, 10)), np.ones(10))

model = [l1, l2, l3, l4, l5, l6, l7, l8]
data = (tr_data, tr_label, vl_data, vl_label)

network = c.CNN(model, c.SoftmaxLayer, data, lr=0.0001, momentum=0.9, dropout=0.8)
t, v = network.runSGD(10, batch_size=10)
