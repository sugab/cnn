import numpy as np
import helper as h
import mnist
import cnn2 as c

np.random.seed(2)

l1 = c.ConvolutionLayer(h.random((16, 1, 5, 5)), np.ones((16, 1)), p=2, activation=c.LeakyReLULayer)
l2 = c.PoolingLayer(2, 2)
l3 = c.ConvolutionLayer(h.random((32, 16, 3, 3)), np.ones((32, 1)), activation=c.LeakyReLULayer)
l4 = c.PoolingLayer(2, 2)
l5 = c.ConvolutionLayer(h.random((32, 32, 3, 3)), np.ones((32, 1)), p=1, activation=c.LeakyReLULayer)
l6 = c.PoolingLayer(2, 2)
l7 = c.FCLayer(h.random((288, 64)), np.ones(64), activation=c.LeakyReLULayer)
l8 = c.FCLayer(h.random((64, 10)), np.ones(10))

model = [l1, l2, l3, l4, l5, l6, l7, l8]

tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 1000))
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(1000, 1100))

tr_data = tr_data[:, np.newaxis, :, :]
vl_data = vl_data[:, np.newaxis, :, :]

data = [tr_data, tr_label, vl_data, vl_label]

network = c.CNN(model, c.SoftmaxLayer, data, lr=0.00005, momentum=0.9, dropout=0.8)
t, v = network.run(500, 25)

np.save('tubes_lrelu_1_t', t)
np.save('tubes_lrelu_1_v', v)
