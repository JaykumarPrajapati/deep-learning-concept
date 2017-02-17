import tensorflow as tf
sess = tf.Session()

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')         # use mul intead of * ... just to set name

print '\n\n===================creating graph=========================='
summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)      # create directory with name 'log_simple_graph'
                                                                                # $ tensorboard --logdir=log_simple_graph

print '\n\n======================loss===================='
y_ = tf.constant(0.0)
loss = (y - y_)**2
print "loss:", loss

print '\n\n=======================gradient================'
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
print "gradient decent:", optim

grad_and_vars = optim.compute_gradients(loss)
sess.run(tf.global_variables_initializer())
print '\n\n=========================print first gradient value========================'
g1 = sess.run(grad_and_vars[0][1])
print grad_and_vars
print g1

print '\n\n=============================apply gradient value============================'
sess.run(optim.apply_gradients(grad_and_vars))
updated_w = sess.run(w)
print 'updated_weight:', updated_w

print '\n\n=============================gradient + optimization ============================'
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
# for i in range(100):
#     sess.run(train_step)
#     if i%10==0:
#         print sess.run(y)


print '\n ================================addding output y summary to graph==================='

summary_y = tf.summary.scalar('riddhi_output',y)
for i in range(100):
    sess.run(train_step)
    summary_y_str = sess.run(summary_y)
    summary_writer.add_summary(summary_y_str,i)
    if i%10==0:
        print sess.run(y)