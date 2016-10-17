import numpy as np
import cnn
import cifar

train_batch = cifar.unpickle('cifar-10-batches-py/data_batch_1')
train_data = train_batch['data']
train_label = train_batch['labels']

validation_batch = cifar.unpickle('cifar-10-batches-py/data_batch_2')
validation_data = validation_batch['data']
validation_label = validation_batch['labels']

np.random.seed(1)

syn1 = np.random.normal(0, 0.01, (3072, 1000))
b1 = np.ones(1000)
syn2 = np.random.normal(0, 0.01, (1000, 10))
b2 = np.ones(10)

ldsyn1 = 0
ldsyn2 = 0

lr = 0.000001
m = 0.9
mb_size = 100

tr_result = np.zeros((50, 100))
vl_result = np.zeros((50, 2))

for i in xrange(50):

    for  j in xrange(len(train_data) / mb_size):

        l0 = train_data[(j*mb_size):((j+1)*mb_size)]
        label = train_label[(j*mb_size):((j+1)*mb_size)]

        l1 = cnn.relu(np.dot(l0, syn1) + b1)
        l2_sum = np.dot(l1, syn2) + b2

        l2, loss = cnn.forward_softmax(l2_sum, label)

        print "EPOCH %s BATCH %s LOSS = %s" % (i, j, loss)
        tr_result[i, j] = loss

        ld2 = cnn.backward_softmax(l2, label)
        ld1 = np.dot(ld2, syn2.T) * cnn.relu(l1, deriv=True)

        dsyn2 = l1.T.dot(ld2) + ldsyn2 * m
        dsyn1 = l0.T.dot(ld1) + ldsyn1 * m

        syn2 -= dsyn2 * lr
        syn1 -= dsyn1 * lr

        b2 -= np.sum(ld2, axis=0) * lr
        b1 -= np.sum(ld1, axis=0) * lr

        ldsyn2 = dsyn2
        ldsyn1 = dsyn1

    l1 = cnn.relu(np.dot(validation_data, syn1) + b1)
    l2_sum = np.dot(l1, syn2) + b2
    l2, loss = cnn.forward_softmax(l2_sum, validation_label)
    acc = np.sum((np.argmax(l2, axis=1) == validation_label) * 1) / float(len(validation_label))

    print "EPOCH %s V LOSS = %s V ACC = %s" % (i, loss, acc)
    vl_result[i, 0] = loss
    vl_result[i, 1] = acc

np.save('tr_result.npy', tr_result)
np.save('vl_result.npy', vl_result)
