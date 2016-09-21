import numpy as np
import mnist
import bc

#%% Load Data
tr_data, tr_label = mnist.load_mnist(path="", selection=slice(0, 500))
vl_data, vl_label = mnist.load_mnist(path="", selection=slice(1000, 1100))

l0 = tr_data.reshape((tr_data.shape[0], 784))

np.random.seed(1)
bc.Layer.learning_rate = 0.01

l1 = bc.FCLayer((784, 196), activation=bc.SigmoidLayer(), auto_update=True)
l2 = bc.FCLayer((196, 49), activation=bc.SigmoidLayer(), auto_update=True)
l3 = bc.FCLayer((49, 10), auto_update=True)

#%% Training
for i in xrange(450):
    l1_out = l1.forward(l0)
    l2_out = l2.forward(l1_out)
    l3_out = l3.forward(l2_out)

    probs, loss = bc.SoftmaxLayer.forward(l3_out, tr_label)
    result = np.argmax(probs, axis=1)

    error3 = bc.SoftmaxLayer.backward(probs, tr_label)
    error2, _ = l3.backward(error3)
    error1, _ = l2.backward(error2)
    l1.backward(error1)

# print result

#%% Training Check
cft = np.zeros((10, 10))
for j in xrange(len(tr_label)):
    cft[result[j], tr_label[j]] += 1

print "Training Data EPOCH: %s = %s" % (i + 1, np.sum(np.diag(cft)))

#%% Validation Check
l0 = vl_data.reshape((vl_data.shape[0], 784))

l1_out = l1.forward(l0)
l2_out = l2.forward(l1_out)
l3_out = l3.forward(l2_out)

probs, loss = bc.SoftmaxLayer.forward(l3_out, vl_label)
result = np.argmax(probs, axis=1)

cfv = np.zeros((10, 10))
for i in xrange(len(vl_label)):
    cfv[result[i], vl_label[i]] += 1

print "Validation Accuracy: %s" % np.sum(np.diag(cfv))
