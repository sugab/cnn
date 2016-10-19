import numpy as np
import cnn
import cifar

train_batch = cifar.unpickle('cifar-10-batches-py/data_batch_1')
train_data = train_batch['data']
train_data = train_data.reshape((len(train_data), 32, 32, 3), order='F')
train_label = train_batch['labels']

validation_batch = cifar.unpickle('cifar-10-batches-py/data_batch_2')
validation_data = validation_batch['data']
validation_data = validation_data.reshape((len(train_data), 32, 32, 3), order='F')
validation_label = validation_batch['labels']

np.random.seed(1)

f1 = np.random.normal(0, 0.01, (16, 3, 3, 3))
b1 = np.ones(16)
syn3 = np.random.normal(0, 0.01, (3600, 10))
b3 = np.ones(10)

ldf1 = 0
ldsyn3 = 0

lr = 0.0001
m = 0.9
mb_size = 10

tr_result = np.zeros((50, 1000))
vl_result = np.zeros((50, 2))

for i in xrange(25):

    for  j in xrange(len(train_data) / mb_size):

        l0 = train_data[(j*mb_size):((j+1)*mb_size)]
        label = train_label[(j*mb_size):((j+1)*mb_size)]

        l1 = cnn.forward_conv(l0, f1, b1)
        l2, l2_sw = cnn.forward_pool(l1, 2, 2)

        l3_in = l2.reshape((l2.shape[0], l2.shape[1] * l2.shape[2] * l2.shape[3]))

        l3_sum = np.dot(l3_in, syn3) + b3
        l3, loss = cnn.forward_softmax(l3_sum, label)

        print "EPOCH %s BATCH %s LOSS = %s" % (i, j, loss)
        tr_result[i, j] = loss

        ld3 = cnn.backward_softmax(l3, label)
        ld2_out = np.dot(ld3, syn3.T)

        ld2_r = ld2_out.reshape((l2.shape[0], l2.shape[1], l2.shape[2], l2.shape[3]))

        ld2 = cnn.backward_pool(ld2_r, l2_sw, 2, 2)
        ld1, df1, db1 = cnn.backward_conv(l0, l1, ld2, f1)

        dsyn3 = l3_in.T.dot(ld3) + ldsyn3 * m
        syn3 -= dsyn3 * lr
        b3 -= np.sum(ld3, axis=0) * lr
        ldsyn3 = dsyn3

        df1 += ldf1 * m
        f1 -= df1 * lr
        b1 -= db1
        ldf1 = df1

    l1 = cnn.forward_conv(validation_data, f1, b1)
    l2, l2_sw = cnn.forward_pool(l1, 2, 2)
    l3_in = l2.reshape((l2.shape[0], l2.shape[1] * l2.shape[2] * l2.shape[3]))
    l3_sum = np.dot(l3_in, syn3) + b3
    l3, loss = cnn.forward_softmax(l3_sum, validation_label)
    acc = np.sum((np.argmax(l3, axis=1) == validation_label) * 1) / float(len(validation_label))

    print "EPOCH %s V LOSS = %s V ACC = %s" % (i, loss, acc)
    vl_result[i, 0] = loss
    vl_result[i, 1] = acc

np.save('tr_result_cnn.npy', tr_result)
np.save('vl_result_cnn.npy', vl_result)
