#ESAMOS PROBLEMOS:
    #Paleidžiant daug kartų training metodą, atsiranda ResourceExhaustedError. Taigi, reikia išsiaiškinti kaip atlaisvinti GPU atmintį po individo
    #treniravimo
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Nerodo warningu, tik tai ką atspausdini su print ir Errorus.
import tensorflow as tf
import math
import time


class CNN:
    Print_intermediate_accuracy = 100
    tf.set_random_seed(1)

    def __init__(self, pop, IterationCountTrain, BatchSizeTrain, BatchSizeTest, TimeLimit, data_set):
        self.IterationCountTrain = IterationCountTrain
        self.BatchSizeTrain = BatchSizeTrain
        self.TimeLimit = TimeLimit
        self.BatchSizeTest = BatchSizeTest
        self.graph = tf.Graph() #Sukuriu grafą į kurį kelsiu arhitektūros modelį.
        self.mnist = data_set
        with self.graph.as_default():
            self.sess = tf.InteractiveSession()
            self.pop = pop #Populiacija yra saugoma CNN objekte
            self.x = tf.placeholder(tf.float32, shape=[None, 784]) #Čia bus saugomi paveikslėlių batch
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10]) #Teisingi prediction
            self.is_train = tf.placeholder(tf.bool) #Naudojamas nustatyti ar vykdomas testavimas. Tam, kad galeciau nevykdyti drop_out

    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x, ksize):
      return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                            strides=[1, ksize, ksize, 1], padding='SAME')

    def avg_pool_2x2(self, x, ksize):
      return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                            strides=[1, ksize, ksize, 1], padding='SAME')

    #Dinamiškai sukuriu CNN modelį pagal duotą arhitektūrą
    def CNN_model(self, data, LA):
        with self.graph.as_default():
            currentSize = 784 #Paveikslėlų išmatavimas - 28x28
            currentDepth = 1 #Nes MNIST paveikslėliai yra Black/Wite (binary)
            squareShape = int(math.sqrt(currentSize)) #Reikalingas, kad išlaikyčiau informaciją apie einamas paveikslėlio dimensijas
            x_image = tf.reshape(data, [-1, 28, 28, 1])
            # Dar nežinau kaip padaryti, kad metodas žinotu kada vykdomas testas.

            firstFC = True
            for i in range(len(LA)):
                if LA[i][0] == "Conv":
                    kSize = LA[i][1]
                    depth = LA[i][2]

                    W_conv = self.weight_variable([kSize, kSize, currentDepth, depth])
                    b_conv = self.bias_variable([depth])

                    x_image = tf.nn.relu(self.conv2d(x_image, W_conv) + b_conv)
                    currentDepth = depth

                elif LA[i][0] == "APool" or LA[i][0] == "MPool":
                    kSize = LA[i][1]
                    if LA[i][0] == "MPool":
                        x_image = self.max_pool_2x2(x_image, kSize)
                    else:
                        x_image = self.avg_pool_2x2(x_image, kSize)
                    squareShape = math.ceil(squareShape / kSize)

                elif LA[i][0] == "FC":
                    out = LA[i][1]
                    drop = LA[i][2]
                    if firstFC: #Šitą vėliau pakeisiu.
                        W_fc = self.weight_variable([squareShape * squareShape * currentDepth, out])
                        h_fc = tf.reshape(x_image, [-1, squareShape * squareShape * currentDepth])
                        firstFC = False
                    else:
                        W_fc = self.weight_variable([LA[i-1][1], out])

                    b_fc = self.bias_variable([out])

                    h_fc = tf.nn.relu(tf.matmul(h_fc, W_fc) + b_fc)

                    h_fc = tf.cond(self.is_train, lambda: tf.nn.dropout(h_fc, drop), lambda: h_fc)#Jei treniruojame, tada nustatome drop_out pagal duotą parametrą. Jei ne tada drop_out nededame.


            #Toliau tiesiog prisegu FC su 10 output
            flat = h_fc

            W_fc = self.weight_variable([out, 10])
            b_fc = self.bias_variable([10])

            y_conv = tf.matmul(flat, W_fc) + b_fc

            return y_conv

    def trainCNN(self, LA):

        #Sukuriu grafui modelį
        y_conv = self.CNN_model(self.x, LA)

        #Apskaičiuojam prediction(cross_entropy, t.t.)
        with self.graph.as_default():
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        result = 0
        print("Training has started...")
        #Paleidžiu sessioną kuris treniruos ir testuos sukurtą grafo modelį
        #1
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            t0 = time.clock()

            for i in range(self.IterationCountTrain):
                try:
                    batch = self.mnist.train.next_batch(self.BatchSizeTrain)
                    if (i + 1) % self.Print_intermediate_accuracy == 0:
                        # Paduodu į grafą paveikslėlių batchus su teisingais label, gražina batch accuracy
                        train_accuracy = accuracy.eval(
                            feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                        print('step %d, training accuracy %g' % (i + 1, train_accuracy))
                    if time.clock() - t0 > self.TimeLimit:
                        print("Exceeded training time limit Accuracy = ", 0)
                        a = 100
                        b = 2
                        break
                    train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                except Exception as error:
                    print('Caught this error: ' + repr(error))
                    print("fuuu")

            if time.clock() - t0 > self.TimeLimit:
                pass

            # Testavimas (dėl problemų su memory išskaidau 10k paveiksleliu į bachus)
            for i in range(self.BatchSizeTest):
                batch = self.mnist.test.next_batch(int(math.floor(10000 / self.BatchSizeTest)))
                try:
                    temp = accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: False})
                except Exception as error:
                    print('Caught this error: ' + repr(error))
                    print("fuuu")
                result += temp

        tf.reset_default_graph()
        #2
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            t0 = time.clock()

            for i in range(self.IterationCountTrain):
                try:
                    batch = self.mnist.train.next_batch(self.BatchSizeTrain)
                    if (i + 1) % self.Print_intermediate_accuracy == 0:
                        # Paduodu į grafą paveikslėlių batchus su teisingais label, gražina batch accuracy
                        train_accuracy = accuracy.eval(
                            feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                        print('step %d, training accuracy %g' % (i + 1, train_accuracy))
                    if time.clock() - t0 > self.TimeLimit:
                        print("Exceeded training time limit Accuracy = ", 0)
                        a = 100
                        b = 2
                        break
                    train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                except Exception as error:
                    print('Caught this error: ' + repr(error))
                    print("fuuu")
                    return 0

            if time.clock() - t0 > self.TimeLimit:
                pass

            # Testavimas (dėl problemų su memory išskaidau 10k paveiksleliu į bachus)
            for i in range(self.BatchSizeTest):
                batch = self.mnist.test.next_batch(int(math.floor(10000 / self.BatchSizeTest)))
                try:
                    temp = accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: False})
                except:
                    print("Error, most likely memory leakage")
                    return 0
                result += temp
        tf.reset_default_graph()
        #3
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            t0 = time.clock()
            a = 0
            while a < 5:
                for i in range(self.IterationCountTrain):
                    try:
                        batch = self.mnist.train.next_batch(self.BatchSizeTrain)
                        if (i + 1) % self.Print_intermediate_accuracy == 0:
                            # Paduodu į grafą paveikslėlių batchus su teisingais label, gražina batch accuracy
                            train_accuracy = accuracy.eval(
                                feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                            print('step %d, training accuracy %g' % (i + 1, train_accuracy))
                        if time.clock() - t0 > self.TimeLimit:
                            print("Exceeded training time limit Accuracy = ", 0)
                            a = 100
                            b = 2
                            break
                        train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: True})
                    except Exception as error:
                        print('Caught this error: ' + repr(error))
                        return 0

                if time.clock() - t0 > self.TimeLimit:
                    pass

                # Testavimas (dėl problemų su memory išskaidau 10k paveiksleliu į bachus)
                for i in range(self.BatchSizeTest):
                    batch = self.mnist.test.next_batch(int(math.floor(10000 / self.BatchSizeTest)))
                    try:
                        temp = accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: False})
                    except:
                        print("Error, most likely memory leakage")
                        return 0
                    result += temp
                if time.clock() - t0 > self.TimeLimit:
                    break

                should_be = 0
                if a == 0:
                    should_be = 93.3
                    result /= 3
                if a == 1:
                    should_be = 95.6
                if a == 2:
                    should_be = 96.5
                if a == 3:
                    should_be = 97.4
                a += 1

                print((result / self.BatchSizeTest)*100)
                if (result / self.BatchSizeTest)*100 < should_be:
                    break
                result = 0

            result = 0
            #Testavimas (dėl problemų su memory išskaidau 10k paveiksleliu į bachus)
            for i in range(self.BatchSizeTest):
                batch = self.mnist.test.next_batch(int(math.floor(10000 / self.BatchSizeTest)))
                try:
                    temp = accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.is_train: False})
                except:
                    print("Error, most likely memory leakage")
                    return 0
                result += temp

            print('Accuracy = ' + str("%.4f" % (result / self.BatchSizeTest)) + ' Total computation time = ' + str("%.2f" % (time.clock() - t0)) + "s")

        return result / self.BatchSizeTest

    # Pagrindinis metodas individo treniravimui
    def exec_cnn(self):
        result = self.trainCNN(self.pop) * 100
        tf.reset_default_graph()
        return result

