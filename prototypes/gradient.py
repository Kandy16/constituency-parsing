import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

w1 = tfe.Variable(initial_value=tf.ones(shape=(5,5)))
w2 = tfe.Variable(initial_value=5*tf.ones(shape=(5,5)))
#print(w1)
#print(w2)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1000)

with tf.GradientTape() as tape:
    prod1 = w1*w2
    prod2 = w2*w1
    dw = tape.gradient([prod2], [w1,w2])

print(dw)
grad_op = optimizer.apply_gradients(zip(dw,[w1,w2]))

print(w1)
print(w2)

#optimizer.apply_gradients([(dw[0],w1)])

#print(w1)
#print(w2)
