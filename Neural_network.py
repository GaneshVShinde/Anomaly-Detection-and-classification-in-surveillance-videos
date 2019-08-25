import tensorflow as tf
from tensorflow.contrib.layers import fully_connected,dropout,l2_regularizer

x = tf.placeholder('float', [None, 4096])
y = tf.placeholder('float')
hl = tf.placeholder('float') #hinge_loss


n_hidden1 = 512
n_hidden2 = 32
n_output = 1
batch_size = 32


def dnn_network(X):
    tf_h1 = {'weights':tf.Variable(tf.random_normal([4096, n_hidden1])), #4096 feature vector
                      'biases':tf.Variable(tf.random_normal([n_hidden1]))}

    tf_h2 = {'weights':tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
                      'biases':tf.Variable(tf.random_normal([n_hidden2]))}

    tf_op = {'weights':tf.Variable(tf.random_normal([n_hidden2, n_output])),
                    'biases':tf.Variable(tf.random_normal([n_output])),}


    l1 = tf.add(tf.matmul(X,tf_h1['weights']), tf_h1['biases'])
    l1 = tf.nn.relu(l1)
    l1 = tf.layers.dropout(l1, 0.6)
    
    
    l2 = tf.add(tf.matmul(l1,tf_h2['weights']), tf_h2['biases'])
    l2  = tf.layers.dropout(l2, 0.6)
    
    
    output = tf.matmul(l2,tf_op['weights']) + tf_op['biases']
    output = tf.nn.sigmoid(output)
    

    return output , tf_h1['weights'] , tf_h2['weights'] , tf_op['weights']


