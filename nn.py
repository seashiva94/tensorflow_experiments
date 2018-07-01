from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
num_steps = 50000
batch_size = 128
display_step = 100

hidden_1 = 256
hidden_2 = 256
num_input = 784
num_classes = 10

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, hidden_1])),
    'h2': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
    'out': tf.Variable(tf.random_normal([hidden_2, num_classes])),
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden_1])),
    'b2': tf.Variable(tf.random_normal([hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes])),

}

def nn(x):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer3 = tf.matmul(layer2, weights['out']) + biases['out']
    return layer3


logits = nn(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = Y))
print("defining loss_op")

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
print("defining optimizer")

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("defined accuracy")

init = tf.global_variables_initializer()
print("initialized")

with tf.Session() as sess:
    sess.run(init)
    print("tranining")
    for step in range(num_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict = {X: batch_x, Y: batch_y})
        if step % display_step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: batch_x, Y: batch_y})
            print("step: "+ str(step) + " loss: " + str(loss) + "acc :" + str(acc))
    print("traning complete")
    print("testing")
    tacc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

    print("acc : " + str(tacc))
