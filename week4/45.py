import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

epochs = 60000

X_train = np.loadtxt('wine/wine_X_train.txt', dtype=float)
X_test  = np.loadtxt('wine/wine_X_test.txt', dtype=float)
t_train = np.loadtxt('wine/wine_t_train.txt', dtype=float)
t_test  = np.loadtxt('wine/wine_t_test.txt', dtype=float)

#Shuffle training set
sampls = random.sample(range(len(X_train)), len(X_train))
X_train = X_train[sampls]
t_train = t_train[sampls]

mean1 = np.mean(X_train[:,0])
mean2 = np.mean(X_train[:,1])

sd1 = np.std(X_train[:,0])
sd2 = np.std(X_train[:,1])

X_train[:,0] = (X_train[:,0] - mean1)/sd1
X_train[:,1] = (X_train[:,1] - mean2)/sd2

X_test[:,0] = (X_test[:,0] - mean1)/sd1
X_test[:,1] = (X_test[:,1] - mean2)/sd2

#For plotting
xx, yy = np.meshgrid(np.linspace(min(X_train[:,0]), max(X_train[:,0]), 500), np.linspace(min(X_train[:,1]), max(X_train[:,1]), 1000))

x = tf.placeholder(tf.float32, [None, 2])
t = tf.placeholder(tf.float32, [None, 3])

W = tf.get_variable("W", [2, 3], initializer=tf.random_normal_initializer)
b = tf.get_variable("b", [3], initializer=tf.random_normal_initializer)

#a = xW + b

a = tf.matmul(x, W) + b
y = tf.nn.softmax(a)

prediction = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(t, 1)), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=a))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for epoch in range(epochs):
        # _, loss_value = session.run([optimizer, loss], feed_dict={x: X_train, t: t_train})
        for j in range(0, len(X_train), 32):
            if j+32 <= len(X_train):
                pass #_, loss_value = session.run([optimizer, loss], feed_dict={x: X_train[j:j+32], t: t_train[j:j+32]})
            else:
                _, loss_value = session.run([optimizer, loss], feed_dict={x: X_train[j:], t: t_train[j:]})

        if epoch % 10000 == 0:
            print("Epoch: {} \t Loss: {}".format(epoch, loss_value))
    
    print("Done")

    y_test = session.run(prediction, feed_dict={x: X_test})
    y_plot1 = session.run(prediction, feed_dict={x: np.c_[xx.ravel(), yy.ravel()]})

    acc = session.run(accuracy, feed_dict={x: X_test, t:t_test})
    print("Accuracy: {}".format(acc))


W1 = tf.get_variable("W1", [2, 5], initializer=tf.random_normal_initializer)
b1 = tf.get_variable("b1", [5], initializer=tf.random_normal_initializer)   
W2 = tf.get_variable("W2", [5, 3], initializer=tf.random_normal_initializer)
b2 = tf.get_variable("b2", [3], initializer=tf.random_normal_initializer)

a1 = tf.matmul(x, W1) + b1
z1 = tf.nn.relu(a1)
a2 = tf.matmul(z1, W2) + b2
y = tf.nn.softmax(a2)

prediction = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(t, 1)), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=a2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init)


    for epoch in range(epochs):

        if epoch % 1000 == 0:
            sampls = random.sample(range(len(X_train)), len(X_train))
            X_train = X_train[sampls]
            t_train = t_train[sampls]

        # _, loss_value = session.run([optimizer, loss], feed_dict={x: X_train, t: t_train})
        for j in range(0, len(X_train), 32):
            if j+32 <= len(X_train):
                _, loss_value = session.run([optimizer, loss], feed_dict={x: X_train[j:j+32], t: t_train[j:j+32]})
            else:
                _, loss_value = session.run([optimizer, loss], feed_dict={x: X_train[j:], t: t_train[j:]})

        if loss_value < 0.1:
            print(loss_value)
            break
        
        if epoch % 10000 == 0:
            print("Epoch: {} \t Loss: {}".format(epoch, loss_value))
    print("Done")

    y_test = session.run(prediction, feed_dict={x: X_test})
    y_plot2 = session.run(prediction, feed_dict={x: np.c_[xx.ravel(), yy.ravel()]})

    acc = session.run(accuracy, feed_dict={x: X_test, t:t_test})
    print("Accuracy: {}".format(acc))


f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')

y_plot1 = y_plot1.reshape(xx.shape)
ax1.contourf(xx, yy, y_plot1)
ax1.scatter(X_train[t_train[:,0]==1, 0], X_train[t_train[:,0]==1, 1], label="Barolo")
ax1.scatter(X_train[t_train[:,1]==1, 0], X_train[t_train[:,1]==1, 1], label="Grignolino")
ax1.scatter(X_train[t_train[:,2]==1, 0], X_train[t_train[:,2]==1, 1], label="Barbera")

y_plot2 = y_plot2.reshape(xx.shape)
ax2.contourf(xx, yy, y_plot2)
ax2.scatter(X_train[t_train[:,0]==1, 0], X_train[t_train[:,0]==1, 1], label="Barolo")
ax2.scatter(X_train[t_train[:,1]==1, 0], X_train[t_train[:,1]==1, 1], label="Grignolino")
ax2.scatter(X_train[t_train[:,2]==1, 0], X_train[t_train[:,2]==1, 1], label="Barbera")
ax2.legend()

plt.show()