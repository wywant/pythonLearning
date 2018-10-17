from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
sess = tf.Session()
print(sess.run(total))
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

# references
# https://www.tensorflow.org/guide/low_level_intro
# tensorboard --logdir .

