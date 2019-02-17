import tensorflow as tf
import numpy as np
import os
import random
import json
import matplotlib.pyplot as plt

class UNET():
  def __init__(self, trainDB, devDB, testDB, trainMeta, devMeta, testMeta, dirname, hparameters=None):

    self.filterNum = list(map(lambda x: int(hparameters["filterNum0"] * hparameters["filterBase"]**x), range(5)))
    self.hparameters = hparameters
    self.trainMeta = trainMeta
    self.devMeta = devMeta
    self.testMeta = testMeta

    dataShape = trainMeta[0]["shape"]
    self.createGraph(trainDB, devDB, testDB, dataShape)

    self.dirname = dirname

  def createGraph(self, trainDB, devDB, testDB, shape):
    self.shapes = []

    # Database iterator and operations to reinitialize the iterator
    iter = tf.data.Iterator.from_structure(trainDB.output_types, trainDB.output_shapes)
    self.train_init = iter.make_initializer(trainDB)
    self.dev_init = iter.make_initializer(devDB)
    self.test_init = iter.make_initializer(testDB)

    # Input layer
    self.xImgs, self.yLabels = iter.get_next()

    # self.x = tf.placeholder(tf.float32, shape=[None] + shape, name="dev_loss_summary")
    self.x = self.xImgs
    self.y = self.yLabels

    self.shapes.append(tf.shape(self.x))

    # Learning rate
    self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")

    # Global learning step counter (Used for learning rate decay calculation)
    gs = tf.Variable(0, trainable=False)

    A = self.x

    convs = []
    actFun = tf.nn.leaky_relu
#     actFun = tf.nn.relu

    with tf.variable_scope("Conv0"):
#       convs.append(A)
      A = tf.layers.conv2d(A, self.filterNum[0], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[0], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))
      convs.append(A)
      A = tf.layers.dropout(A)
      A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")

    with tf.variable_scope("Conv1"):
#       convs.append(A)
      A = tf.layers.conv2d(A, self.filterNum[1], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[1], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))
      convs.append(A)
      A = tf.layers.dropout(A)
      A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")

    with tf.variable_scope("Conv2"):
#       convs.append(A)
      A = tf.layers.conv2d(A, self.filterNum[2], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[2], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))
      convs.append(A)
      A = tf.layers.dropout(A)
      A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")

    with tf.variable_scope("Conv3"):
      convs.append(A)
      A = tf.layers.conv2d(A, self.filterNum[3], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[3], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))
#       convs.append(A)
      A = tf.layers.dropout(A)
      A = tf.layers.max_pooling2d(A, (2,2), (2,2), padding="SAME")

    with tf.variable_scope("ConvMiddle"):
      A = tf.layers.conv2d(A, self.filterNum[4], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[4], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))
      A = tf.layers.dropout(A)

    with tf.variable_scope("Upscale3"):
      A = tf.image.resize_images(A, size=self.shapes[4][1:3] )
      A = tf.layers.conv2d(A, self.filterNum[3], (2, 2), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.concat([A, convs.pop()] , 3)
      A = tf.layers.conv2d(A, self.filterNum[3], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[3], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))

    with tf.variable_scope("Upscale2"):
      A = tf.image.resize_images(A, size=self.shapes[3][1:3] )
      A = tf.layers.conv2d(A, self.filterNum[2], (2, 2), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.concat([A, convs.pop()] , 3)
      A = tf.layers.conv2d(A, self.filterNum[2], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[2], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))

    with tf.variable_scope("Upscale1"):
      A = tf.image.resize_images(A, size=self.shapes[2][1:3] )
      A = tf.layers.conv2d(A, self.filterNum[1], (2, 2), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.concat([A, convs.pop()] , 3)
      A = tf.layers.conv2d(A, self.filterNum[1], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[1], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))

    with tf.variable_scope("Upscale0"):
      A = tf.image.resize_images(A, size=self.shapes[1][1:3] )
      A = tf.layers.conv2d(A, self.filterNum[0], (2, 2), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.concat([A, convs.pop()] , 3)
      A = tf.layers.conv2d(A, self.filterNum[0], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      A = tf.layers.conv2d(A, self.filterNum[0], (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(A))

    with tf.variable_scope("Out"):
      A = tf.layers.conv2d(A, 2, (3, 3), (1,1), padding="SAME", activation=actFun, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.out = tf.layers.conv2d(A, 1, (1, 1), (1,1), padding="SAME", activation=tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.hparameters["lam"]))
      self.shapes.append(tf.shape(self.out))

    # Loss and loss optimizer operations
    # self.loss = tf.losses.sigmoid_cross_entropy(self.out, self.y)# + tf.losses.get_regularization_loss()
    self.loss = -tf.reduce_mean(self.y*tf.log(self.out + 0.0001) + (1-self.y) * tf.log(1 - self.out + 0.0001)) + tf.losses.get_regularization_loss()
    self.learning_rate = tf.train.exponential_decay(self.lr, gs, self.hparameters["alphaTau"], 0.1)

    optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
    self.optimize = optimizer.minimize(self.loss, global_step = gs)

    # Loss summaries
    with tf.name_scope("loss_data"):
      self.devLoss = tf.placeholder(tf.float32, shape=None, name="dev_loss_summary")
      self.trainLoss = tf.placeholder(tf.float32, shape=None, name="train_loss_summary")
      devLossSummary = tf.summary.scalar("DevLoss", self.devLoss)
      trainLossSummary = tf.summary.scalar("TrainLoss", self.trainLoss)
      logDevLossSummary = tf.summary.scalar("DevLossLog", tf.log(self.devLoss))
      logTrainLossSummary = tf.summary.scalar("TrainLossLog", tf.log(self.trainLoss))
    lossSummaries = tf.summary.merge([trainLossSummary, devLossSummary, logTrainLossSummary, logDevLossSummary])

    self.summaries=tf.summary.merge([lossSummaries])
    self.saver = tf.train.Saver()

  def train(self,  niter=1000, batchsize=2, display=False, restart=True, printLoss=True):
#     config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    dirname = self.dirname

    # Make directory for outputing result info
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    # Recording hyperparamters used for the model
    with open(os.path.join(dirname,"hparameters"), "w") as f:
      f.write(json.dumps(self.hparameters))

    # Writer for tensorboard information
    summ_writer = tf.summary.FileWriter(dirname)

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      # Load previous session
      if not restart:
        self.saver.restore(sess, os.path.join(dirname,"model"))

      # Display network architectue
      if printLoss:
        sess.run(self.train_init)
        print("Architecture:")
        for shape in sess.run(self.shapes):
          print(shape)
        # input("Press any key to continue...")

      # Main loop
      for epoch in range(niter):
        if printLoss:
          print(epoch)

        # Run one epoch of train and calculate average loss on train data
        sess.run(self.train_init)
        finTrain = 0
        for j in range(int(self.trainMeta[0]["length"]/batchsize)):
          _, tmpLoss, pred, labels = sess.run([self.optimize, self.loss, self.out, self.y], feed_dict={self.lr: self.hparameters["alpha0"]})
          finTrain+=tmpLoss

        finTrain/=(self.trainMeta[0]["length"]/batchsize)

        # Calculate loss on dev data
        sess.run(self.dev_init)
        finDev = 0
        finOut = np.zeros(shape=(devMeta[0]["length"],) + devMeta[0]["shape"])
        finY = np.zeros(shape=(devMeta[0]["length"],) + devMeta[0]["shape"])
        for j in range(self.devMeta[0]["length"]):
          tmp, out, y = sess.run([self.loss, self.out, self.y])
          finOut[j, :, :] = out
          finY[j, :, :] = y
          finDev+=tmp
        finDev/=self.devMeta[0]["length"]
        tresh = 0.5
        finOut = finOut>=tresh
        truePos = np.sum(finY[finOut == 1] == 1) / np.sum(finY==1)
        falsePos = np.sum(finY[finOut == 1] == 0) / np.sum(finY==0)
        falseNeg = np.sum(finY[finOut==0] == 1) / np.sum(finY==1)
        trueNeg = np.sum(finY[finOut==0] == 0) / np.sum(finY==0)
        accuracy = np.sum(finOut == finY)/np.prod(finY.shape)
        precision = np.sum(finY[finOut])/ np.sum(finOut)
        recall = np.sum(finY[finOut])/ np.sum(finY)
        f1 = 2/(1/precision + 1/recall)

        if np.isnan(finDev):
          return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Results after 1 epoch of training
        if printLoss:
          print("Train: ", finTrain)
          print("Dev: ", finDev)
          print("Acc: ", accuracy)
          print("f1: ", f1)
          print("True Pos: ", truePos, ", False Pos: ", falsePos)

        # Write summaries for tensorboard
#         summ = sess.run(self.summaries, feed_dict={ self.trainLoss: finTrain, self.devLoss: finDev })
#         summ_writer.add_summary(summ, epoch)

        # Write results to a file
        with open(os.path.join(dirname,"result"), "a") as f:
          f.write(json.dumps({'finDevLoss': float(finDev), 'finTrainLoss':float(finTrain)}))

      # Save final model parameters
      self.saver.save(sess, os.path.join(dirname,"model"))

      # Displaying original and reconstructed images for visual validation
      if display:
        sess.run(self.dev_init)
        for i in range(10):
          x, y, out = sess.run([self.x, self.y, self.out])
#           print(x.shape, y.shape, out.shape)
#           print(np.max(x), np.max(y), np.max(out))
          plt.imshow(x[0])
          plt.show()
          plt.imshow(y[0].reshape(y[0].shape[:2]), cmap=plt.cm.gray)
          plt.show()
          plt.imshow(out[0].reshape(out[0].shape[:2]), cmap=plt.cm.gray)
          plt.show()
          
          for j in range(1,5):
            tresh = 0.2 * j
            truePos = np.sum(y[out>= tresh] == 1)/np.sum(y == 1)
            trueNeg = np.sum(y[out < tresh] == 0)/np.sum(y == 0)
            falsePos = np.sum(y[out>= tresh] == 0)/np.sum(y == 1)
            falseNeg = np.sum(y[out < tresh] == 1)/np.sum(y == 0)
            accuracy = np.sum((out >= tresh) == y)/np.prod(y.shape)
            print(tresh, accuracy)
            print("True Pos: ", truePos, ", Fals Pos: ", falsePos)
            print("Fals Neg: ", falseNeg, ", True Neg: ", trueNeg)

        print("dev done")
        print("starting test")

        sess.run(self.train_init)
        xs, ys, outs = sess.run([self.x, self.y, self.out])
        for i in range(2):
          for x, y, out in zip(xs, ys, outs):
#             print(np.sum(y*np.log(out + 0.0001) + 1-y * np.log(1 - out + 0.0001))/np.prod(y.shape))
            plt.imshow(x)
            plt.show()
            plt.imshow(y.reshape(y.shape[:2]), cmap=plt.cm.gray)
            plt.show()
            plt.imshow(out.reshape(out.shape[:2]), cmap=plt.cm.gray)
            plt.show()
            for j in range(1,5):
              tresh = 0.2 * j
              truePos = np.sum(y[out>= tresh] == 1)/np.sum(y == 1)
              falsePos = np.sum(y[out>= tresh] == 0)/np.sum(y == 1)
              accuracy = np.sum((out >= tresh) == y)/np.prod(y.shape)
              print(tresh, accuracy)
              print("True Pos: ", truePos, ", False Pos: ", falsePos)

    return finDev, finTrain, accuracy, precision, recall, f1
