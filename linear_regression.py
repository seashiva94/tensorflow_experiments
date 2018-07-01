import tensorflow as tf
import numpy as np

data = np.array([1,1],[2,2],[3,3],[4,4])
# defining variables and constants
x = tf.placeholder()
y = tf.placeholder()
m = tf.variable()
b = tf.variable()

# prediction, loss and others
prediction = m*x + b
loss = np.sum(np.square(y - prediction))
optimizer = tf.GradientDescentOptimizer(loss)
session =  tf.Session()

#initialize variables
# iterate over all data points to minimize loss
# find the new value of slope and things

#for i in data.shape[0]:
#    iterate

# do for actual dataset, house prices
# push to github in a tf repo
