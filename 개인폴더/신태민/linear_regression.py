import tensorflow as tf
tf.set_random_seed(2013122141)

x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2001) :
    sess.run(train)
    if i % 20 == 0:
        print(i, sess.run(cost), sess.run(W), sess.run(b))