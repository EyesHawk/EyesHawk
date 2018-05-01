import tensorflow as tf

tf.set_random_seed(2013122141)
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle = False, name = "filename_queue")
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)
train_x_batch, train_y_batch = tf.train.batch([xy[0 : -1], xy[-1 :]], batch_size = 10)
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(hypothesis - Y)
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X : x_batch, Y : y_batch})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\n Prediction :\n", hy_val)
coord.request_stop()
coord.join(threads)
