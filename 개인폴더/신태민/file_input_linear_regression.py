import tensorflow as tf
import numpy as np

tf.set_random_seed(2013122141)
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(hypothesis - Y)
train = tf.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X : x_data, Y : y_data})
    if step % 10 == 0:
        print(step, " Cost : ", cost_val, "\nPrediction : \n", hy_val)

