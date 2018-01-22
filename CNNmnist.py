#ESAMOS PROBLEMOS:
    #Paleidžiant daug kartų training metodą, atsiranda ResourceExhaustedError. Taigi, reikia išsiaiškinti kaip atlaisvinti GPU atmintį po individo
    #treniravimo

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import math

sess = tf.InteractiveSession()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, ksize):
  return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                        strides=[1, ksize, ksize, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
isTest = tf.placeholder(tf.bool)

def CNN_model(data, LA):
    currentSize = 784
    currentDepth = 1
    squareShape = int(math.sqrt(currentSize))
    x_image = tf.reshape(data, [-1, 28, 28, 1])
    # Nežinau kaip padaryti, kad metodas žinotu kada vykdomas testas.


    for i in range(len(LA)):
        if LA[i][0] == "Conv":
            kSize = LA[i][1]
            depth = LA[i][2]

            W_conv = weight_variable([kSize, kSize, currentDepth, depth])
            b_conv = bias_variable([depth])

            x_image = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)

            currentDepth = depth

        elif LA[i][0] == "Pool":
            kSize = LA[i][1]
            x_image = max_pool_2x2(x_image, kSize)
            squareShape = math.ceil(squareShape / kSize)

        elif LA[i][0] == "FC":
            out = LA[i][1]

            W_fc = weight_variable([squareShape * squareShape * currentDepth, out])
            b_fc = bias_variable([out])
            flat = tf.reshape(x_image, [-1, squareShape * squareShape * currentDepth])
            h_fc = tf.nn.relu(tf.matmul(flat, W_fc) + b_fc)


    # Toliau tiesiog prisegu FC su 10 output
    flat = h_fc

    W_fc = weight_variable([out, 10])
    b_fc = bias_variable([10])

    y_conv = tf.matmul(flat, W_fc) + b_fc

    return y_conv

def trainCNN(x, LA):

    y_conv = CNN_model(x, LA)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # Treniravimas
        for i in range(400):
            batch = mnist.train.next_batch(50)
            if i % 200 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], isTest:True})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        result = 0
        # Testavimas (dėl problemų su memory išskaidau 10k paveiksleliu į batchus po 333 paveikslėlius)
        batchSize = 30
        for i in range(batchSize):
            batch = mnist.test.next_batch(int(math.floor(10000 / batchSize)))
            try:
                temp = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            except:
                print("Got em, ERROR, GREIČIAUSIAI MEMORY")
                return 0
            result += temp

        print('Final result = %g' % (result / batchSize))
        del sess
        return (result / batchSize)

# Pagrindinis metodas individo treniravimui
def execCNN(LA):
    return trainCNN(x, LA)





