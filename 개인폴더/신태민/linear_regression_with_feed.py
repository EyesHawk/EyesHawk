import tensorflow as tf
tf.set_random_seed(2013122141)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])
hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2001) :
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1, 2, 3], Y : [1, 2, 3]})
    if i % 20 == 0:
        print(i, cost_val, W_val, b_val)