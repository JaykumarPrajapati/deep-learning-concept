from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

X1 , Y1 = shuffle(X,Y,random_state=0)

X1 = X1.T
y1_array = np.zeros((3,150))
for i in range(150):
    y1_array[Y1[i]][i] =1

import tensorflow as tf

# Set parameters
learning_rate = 0.001
training_iteration = 1000
batch_size = 10
display_step = 2

# TF graph input
x = tf.placeholder("float", [4, X1.shape[1]]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [3, X1.shape[1]]) # 0-9 digits recognition => 10 classes

# Create a model
# Set model weights
W1 = tf.Variable(tf.random_normal([10,4]), name='W1')
W2 = tf.Variable(tf.random_normal([3,10]), name ='W2')
b1 = tf.Variable(tf.random_normal([10,1]), name='b1')
b2 = tf.Variable(tf.random_normal([3,1]), name='b2')


# Construct a linear model
W1x = tf.matmul(W1, x, name= 'wx')  # Softmax
H1 = tf.add(W1x, b1, name='H1')
W2H1 = tf.matmul(W2, H1, name = 'half_score')  # Softmax
score = tf.add(W2H1, b2, name='full_score')
model = tf.nn.softmax(score)

cost_function = -tf.reduce_sum(y*tf.log(model), name='cost_function')
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()

all_sum = []
for value in [W1, W2, H1, cost_function]:
    all_sum.append(tf.summary.histogram(value.op.name, value))
tf.summary.scalar('cost_function', cost_function)
summaries = tf.summary.merge_all()
# summaries = tf.merge_summary(all_sum)


sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats1', sess.graph)

sess.run(tf.global_variables_initializer())

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.

        # Loop over all batches
        for i in range(1):
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: X1, y: y1_array})
            # Compute average loss
            avg_cost += sess.run(cost_function, feed_dict={x: X1, y: y1_array})/10
            summary_writer.add_summary(sess.run(summaries, feed_dict={x: X1, y: y1_array}), iteration)
        # Display logs per eiteration step
        if iteration % display_step == 0:
            # print sess.run(W1)
            print "Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost)

    print "Tuning completed!"


