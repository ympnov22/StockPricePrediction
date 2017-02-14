INPUT = 54
HIDDEN = 50
OUTPUT = 2

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import pandas as pd

start = time.time()

data_x = np.genfromtxt( "PredictionData_x.csv", delimiter=",", filling_values=(0, 0, 0) )

x = tf.placeholder(tf.float32, [None, INPUT])

w1 = tf.Variable(tf.random_normal([INPUT, HIDDEN]))
b1 = tf.Variable(tf.random_normal([HIDDEN]))

wy = tf.Variable(tf.random_normal([HIDDEN, OUTPUT]))
by = tf.Variable(tf.random_normal([OUTPUT]))

h = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.matmul(h, wy) + by

y_ = tf.placeholder(tf.float32, [None, OUTPUT])

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()

ckpt = tf.train.get_checkpoint_state('./')

if(ckpt):
    last_model = ckpt.model_checkpoint_path
    print("load " + last_model)
    saver.restore(sess, last_model)

else: 
    print("no variables")
    exit()

print("predicting...")  
result_y = sess.run(y, feed_dict={x: data_x})
print(result_y[-1])

np.savetxt("PredictionResult.csv", result_y, delimiter=",")

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")