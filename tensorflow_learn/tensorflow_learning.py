import tensorflow as tf

sess = tf.Session()
graph = tf.get_default_graph()
print "without anything: ", graph

input_value = tf.constant(11.0)
print "after input_value: ",graph
graph = tf.get_default_graph()
print "getting default graph: ",graph

operations = graph.get_operations()
print operations
print operations[0].node_def

print '=======================value of constant ======================'
print input_value
print sess.run(input_value)

print '=======================value of variable ======================'
weight = tf.Variable(0.8)           # adding edge to graph..... not node   !!
print graph
for op in graph.get_operations():
    print(op.name)
print '=====================adding multiplication======================'
output_value = weight * input_value
for op in graph.get_operations():
    print(op.name)

print '==========================see all inputs========================='
for op in graph.get_operations():
    print(op.name)
    for op_input in op.inputs:
        print '--',op_input


print '=========================initialise all variables=================='
init = tf.global_variables_initializer()        # not like graph which automatically update on including
                                            # new thing.
                                            # here you have again initialise if you added new variable.
sess.run(init)
print sess.run(output_value)