import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# f, ax = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row')

# ax = np.append(ax[0,:], ax[1,:])

# # (a)
# for i in range(10):
#     tmp = x_train[y_train == i]
#     ax[i].imshow(x_train[y_train == i][0], cmap=plt.cm.gray_r)
# plt.show()

# (b)

# rand_smpls = random.sample(range(len(x_train)), len(x_train)//5)
# smpl_mask = np.zeros(len(x_train))
# smpl_mask[rand_smpls] = 1

# x_val, y_val = x_train[smpl_mask == 1], y_train[smpl_mask == 1]
# x_train, y_train = x_train[smpl_mask == 0], y_train[smpl_mask == 0]

# (c)

n = x_train.shape[1] * x_train.shape[2] # 784

x_train = x_train.reshape(-1, n)
# x_val = x_val.reshape(-1, n)
x_test = x_test.reshape(-1, n)

# (d)

# 1-hot encoding y's
y_train_1h = np.zeros((y_train.shape[0], 10))
y_train_1h[np.arange(y_train.shape[0]), y_train] = 1

# y_val_1h = np.zeros((y_val.shape[0], 10))
# y_val_1h[np.arange(y_val.shape[0]), y_val] = 1

y_test_1h = np.zeros((y_test.shape[0], 10))
y_test_1h[np.arange(y_test.shape[0]), y_test] = 1

#Normalization
# mu = np.mean(x_train)
# sd = np.std(x_train)

# x_train, x_val, x_test = (x_train - mu)/sd, (x_val - mu)/sd, (x_test - mu)/sd

# Shuffle data
smpls = random.sample(range(len(x_train)), len(x_train))
x_train, y_train = x_train[smpls], y_train[smpls]

nodes = 20
epochs = 30
batchsize = 32

i=0
ep = [0]*(epochs)
lo = [0]*(epochs)
acc_train_log = [0]*(epochs)
acc_test_log = [0]*(epochs)

# Construct model
x = tf.placeholder(tf.float32, [None, n])
t = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

#Input layer n (784) dimensions to nodes (20) dimensions
W1 = tf.get_variable("W1", [n, nodes], initializer=tf.random_normal_initializer)
b1 = tf.get_variable("b1", [nodes], initializer=tf.random_normal_initializer)

#Hidden layer 20 to output (10) dimensional
W2 = tf.get_variable("W2", [nodes, 10], initializer=tf.random_normal_initializer)
b2 = tf.get_variable("b2", [10], initializer=tf.random_normal_initializer)

# Defining model
a1 = tf.matmul(x, W1) + b1
z1 = tf.nn.relu(a1)
y = tf.matmul(z1, W2) + b2
# Not applying sofmax to y. Because result is going to be argmaxed in prediction
# and softmaxed in loss

# Prediction and accuracy calc
prediction = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(t, 1)), tf.float32))

# Loss function and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
#loss = tf.reduce_mean(tf.nn.cross_en)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4

with tf.Session(config=config) as session:
    session.run(init)

    print("Starting training.")

    learnR = 0.01
    
    for epoch in range(epochs):

        #Batching
        for j in range(0, len(x_train), batchsize):
            if j+batchsize <= len(x_train):
                _, loss_value = session.run([optimizer, loss], feed_dict={x: x_train[j:j+batchsize], t: y_train_1h[j:j+batchsize], lr: learnR})
            else:
                _, loss_value = session.run([optimizer, loss], feed_dict={x: x_train[j:], t: y_train_1h[j:], lr: learnR})

        if loss_value < 2.3:
            learnR = 0.001
        else:
            learn = 0.01

        ep[epoch] = epoch + 1
        lo[epoch] = loss_value
        acc_train_log[epoch] = session.run(accuracy, feed_dict={x: x_train, t: y_train_1h})
        acc_test_log[epoch] = session.run(accuracy, feed_dict={x: x_test, t: y_test_1h})
        print("Epoch: {} \t Loss: {}".format(epoch+1, loss_value))

    print("Done")

    acc_train = session.run(accuracy, feed_dict={x: x_train, t: y_train_1h})
    acc_test = session.run(accuracy, feed_dict={x: x_test, t: y_test_1h})
    print("Accuracy:\n Train: {}\n Test: {}".format(acc_train, acc_test))

with open('results.txt', 'w') as f:
    f.write("Epoch\tLoss\tTrain\tTest\n")
    for i in range(len(ep)):
        f.write("{}\t{}\t{}\t{}\n".format(ep[i], lo[i], acc_train_log[i], acc_test_log[i]))


f, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1, sharex='col')

ax1.set_title("Loss")
ax1.plot(ep, lo, color='green')
ax2.set_title("Training accuracy")
ax2.plot(ep, acc_train_log, color='red')
ax3.set_title("Test accuracy")
ax3.plot(ep, acc_test_log, color='blue')
plt.show()
