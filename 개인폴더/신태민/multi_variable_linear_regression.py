import tensorflow as tf

tf.set_random_seed(2013122141)
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]
x1 = tf.placeholder(tf.float32, [None])
x2 = tf.placeholder(tf.float32, [None])
x3 = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])
w1 = tf.Variable(tf.random_normal([1]), name = "weight1")
w2 = tf.Variable(tf.random_normal([1]), name = "weight2")
w3 = tf.Variable(tf.random_normal([1]), name = "weight3")
b = tf.Variable(tf.random_normal([1]), name = "bias")
hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict= {x1 : x1_data, x2 : x2_data, x3 : x3_data, Y : y_data})
    if i % 10 == 0:
        print(i, " Cost : ", cost_val, "\nPrediction : ", hy_val, "\n")