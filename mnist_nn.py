import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1,28,28,1])
    conv1 = tf.layers.conv2d(inputs = input_layer,
                         filters = 32,
                         kernel_size = [5,5],
                         padding = "same",
                         activation = tf.nn.relu
                         )
    pool1 = tf.layers.pooling(inputs = conv1, pool_size = [2,2], strides = 2)

def main():
    pass

if __name__ == "__main__":
    tf.app.run()
