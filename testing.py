import os
import json
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from UNET import UNET

import cv2
from sklearn import cluster

colorDiv = 0.1
batchsize = 2

def skip(n, videoIn):
  for i in range(n):
    videoIn.read()

def loadHparams(dir):
  with open(os.path.join(dir,"hparameters"), "r") as f:
    hparams = json.load(f)

  # Fixes
  if "batchsiize" in hparams:
    batchsize = hparams["batchsiize"]
    del hparams["batchsiize"]
    hparams["batchsize"] = batchsize

  if "fshape" in hparams:
    f = hparams["fshape"][0]
    del hparams["fshape"]
    hparams["f"] = f

  if "lambda" in hparams:
    lam = hparams["lambda"]
    del hparams["lambda"]
    hparams["lam"] = lam

  return hparams

def getMetaData():
  with open("train.metadata",'r') as trainf:
    line = trainf.readline()
    metax = json.loads(line)
    line = trainf.readline()
    metay = json.loads(line)
    trainm = ( metax, metay )

  with open("dev.metadata",'r') as devf:
    line = devf.readline()
    metax = json.loads(line)
    line = devf.readline()
    metay = json.loads(line)
    devm = ( metax, metay )

  with open("test.metadata",'r') as testf:
    line = testf.readline()
    metax = json.loads(line)
    line = testf.readline()
    metay = json.loads(line)
    testm = ( metax, metay )

  return trainm, devm, testm

def augment(img, lab):
  outImg = img
  outLab = lab
  # 50% chance to flip image in either direction
  outImg, outLab = tf.cond(tf.random.uniform([]) < 0.5, lambda: (tf.reverse(outImg, axis=[0]), tf.reverse(outLab, axis=[0])) , lambda: (outImg, outLab))
  outImg, outLab = tf.cond(tf.random.uniform([]) < 0.5, lambda: (tf.reverse(outImg, axis=[1]), tf.reverse(outLab, axis=[1])) , lambda: (outImg, outLab))

  # Color Divergance shift
  a = tf.random.normal([1,1,3], mean=1, stddev=colorDiv, dtype=tf.float32)
  outImg = tf.clip_by_value(outImg*a, 0, 1)
  return outImg, outLab

def segKrist(mask, clust):
  xx, yy = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))

  kristali = (mask == 1)
  average = None

  kristaliLoc = np.array([xx, yy])[:,kristali]
  if kristaliLoc.shape[1] > 0: 
    clust.fit(kristaliLoc.T)

    labels = set(clust.labels_)
    average = np.zeros(shape = (2,len(labels)))

    for i in labels:
      a = kristaliLoc[:,clust.labels_==i]
      average[:,i] = a.sum(axis=1)/a.shape[1]

  return average

def rightOrdering(average, oldAverage):
  # Since dbscan asigns different labels to each cluster every iteration,
  # we need to map each cluster to the previous one by proximity
  distance = (np.expand_dims(average.T, axis=3) - oldAverage).transpose([1,0,2])
  norm = np.sum(np.square(distance), axis=0)
  test = np.argmin(norm, axis=0)
  shortDist = np.min(norm, axis=0)<4


  if test[shortDist].size > 0:
    oldAverage[:, shortDist] = average[:,test[shortDist]]
    dodatnoIndeks = ( np.arange(average.shape[1])!=np.expand_dims(test[shortDist], axis=1) ).all(axis=(0))
    oldAverage = np.append(oldAverage, average[:, dodatnoIndeks], axis=1)
  else:
    oldAverage=np.append(oldAverage, average, axis=1)

  mapping = np.arange(oldAverage.shape[1])
  mapping[test] = np.iinfo(mapping.dtype).max
  mapping[:shortDist.size][shortDist] = test[shortDist]
  return average, oldAverage, mapping

def surfun(frame, mask, gmask):
  return np.sum(gmask), gmask


def acquisitionsInit(framecount):
  acquisitions = {
   "surface": {"function": surfun,
              "data": np.zeros(framecount)} 
  }

  return acquisitions

if __name__ == "__main__":
  trainMeta, devMeta, testMeta = getMetaData()

  bytesPerTrainxPoint = np.prod(trainMeta[0]["shape"]) * np.dtype(trainMeta[0]["dtype"]).itemsize
  bytesPerTrainyPoint = np.prod(trainMeta[1]["shape"]) * np.dtype(trainMeta[1]["dtype"]).itemsize
  bytesPerDevxPoint = np.prod(devMeta[0]["shape"]) * np.dtype(devMeta[0]["dtype"]).itemsize
  bytesPerDevyPoint = np.prod(devMeta[1]["shape"]) * np.dtype(devMeta[1]["dtype"]).itemsize
  bytesPerTestxPoint = np.prod(testMeta[0]["shape"]) * np.dtype(testMeta[0]["dtype"]).itemsize
  bytesPerTestyPoint = np.prod(testMeta[1]["shape"]) * np.dtype(testMeta[1]["dtype"]).itemsize

  trainDB = tf.data.FixedLengthRecordDataset("train.data", bytesPerTrainxPoint + bytesPerTrainyPoint)
  devDB = tf.data.FixedLengthRecordDataset("dev.data", bytesPerDevxPoint + bytesPerDevyPoint)
  testDB = tf.data.FixedLengthRecordDataset("test.data", bytesPerTestxPoint + bytesPerTestyPoint)

  def read(bytes, dtype, shapex, shapey):
    inp = tf.decode_raw(bytes, dtype)
    retx = tf.reshape(inp[:np.prod(shapex)], shapex)
    rety = tf.reshape(inp[np.prod(shapex):], shapey)
    return (retx, rety)

  trainDB = trainDB.map(lambda x: read(x, np.dtype(trainMeta[0]["dtype"]),
          trainMeta[0]["shape"], trainMeta[1]["shape"]))
  devDB = devDB.map(lambda x: read(x, np.dtype(devMeta[0]["dtype"]),
          devMeta[0]["shape"], devMeta[1]["shape"]))
  testDB = testDB.map(lambda x: read(x, np.dtype(testMeta[0]["dtype"]),
          testMeta[0]["shape"], testMeta[1]["shape"]))

  trainDB = trainDB.map(augment).shuffle(100).repeat().batch(batchsize)
  devDB = devDB.shuffle(100).repeat().batch(devMeta[0]["length"])
  testDB = testDB.shuffle(100).repeat().batch(testMeta[0]["length"])


  getTest = testDB.make_one_shot_iterator().get_next()
  getDev = devDB.make_one_shot_iterator().get_next()

  hparameters = {}
  hparameters["lam"] = 0.0
  hparameters["filterNum0"] = 64
  hparameters["filterBase"] = 2
  hparameters["alpha0"] = 5*1e-5
  hparameters["alphaTau"] = 43853
  hparameters = loadHparams("model")

  dirname = "model"

  # Setting up shape info
  shape = testMeta[0]["shape"]
  shp = testMeta[0]["shape"][:2]
  x, y = tuple(shp)
  shp = (x, 2*y)


  def acc(y, label, threshold=0.5):
    taken = y>=threshold
    return np.sum(taken==label)/np.prod(label.shape)

  def f1(y, label, threshold=0.5):
    taken = y>=threshold
    P = np.sum(label[taken])/np.sum(taken)
    R = np.sum(label[taken])/np.sum(label)
    return 2/(1/P + 1/R)

  def roc(y, label, threshold=0.5):
    taken = y>=threshold
    TPR = np.sum(label[taken])/np.sum(label)
    FPR = np.sum(taken[label==0])/np.sum(label==0)

    return (FPR, TPR)


  # UNET segmentation model initialisation
  model = UNET(trainDB, devDB, testDB, trainMeta, devMeta, testMeta,
           dirname=dirname , hparameters=hparameters)

  with tf.Session() as sess:
    xs, labels = sess.run(getTest)
    xs = xs[:10]
    labels = labels[:10]
    sess.run(tf.global_variables_initializer())
    model.saver.restore(sess, os.path.join("model", "model"))

    outs = np.zeros(shape=(labels.shape))
    for i, img in enumerate(xs):
      print(i)
      outs[i,:,:,:] =  sess.run(model.out, feed_dict={model.x: img.reshape((1,) + img.shape)})

  # r = np.random.randint(xs.shape[0])

  print(xs.shape)
  print(labels.shape)
  print(outs.shape)

  ts = np.linspace(0,1.01,10000)

  # accs = np.array(list(map( lambda t: acc(outs, labels, t) ,ts)))
  # f1s = np.array(list(map( lambda t: f1(outs, labels, t) ,ts)))
  rocs = np.array(list(map( lambda t: roc(outs, labels, t) ,ts)))
  rocs = sorted(rocs, key = lambda x: x[0])

  # print("Optimal Threshold acc: ", ts[np.argmax(accs)])
  # opt = ts[np.argmax(f1s)]
  # print("Optimal Threshold f1: ", opt, np.max(f1s), np.max(accs) )
  # print("Threshold 0.5: ", 0.5, f1(outs, labels, 0.5), acc(outs, labels, 0.5) )

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  ax.set_title('ROC krivulja', fontdict={'fontsize': 22})
  ax.set_xlabel('FPR', fontsize=20)
  ax.set_ylabel('TPR', fontsize=20)
  ax.set_yticks([0.0, 0.5, 1.0])
  ax.set_yticklabels(["0.0", "0.5", "1.0"])
  ax.set_xticks(np.linspace(0,1,4))

  b, = ax.plot(np.linspace(0,1,4), np.linspace(0,1,4), '--', linewidth=2)
  c, = ax.plot([0,0,1], [0,1,1], '--', linewidth=2)
  a, = ax.plot(*zip(*rocs), linewidth=2)
  a.set_label("UNET")
  b.set_label("Nasumično gađanje")
  c.set_label("Idealan slučaj")
  ax.legend(prop={'size': 18}) 
  plt.rcParams.update({'font.size': 30})
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
    # specify integer or one of preset strings, e.g.
    #tick.label.set_fontsize('x-small') 
    # tick.label.set_rotation('vertical')
  plt.show()


  # ax = fig.add_subplot(1,1,1)

  # ax.set_title('ACC krivulja', fontdict={'fontsize': 22})
  # ax.set_xlabel('Granicna', fontsize=20)
  # ax.set_ylabel('TPR', fontsize=20)
  # ax.set_yticks([0.0, 0.5, 1.0])
  # ax.set_yticklabels(["0.0", "0.5", "1.0"])
  # ax.set_xticks(np.linspace(0,1,4))

  # b, = ax.plot(np.linspace(0,1,4), np.linspace(0,1,4), '--', linewidth=2)
  # c, = ax.plot([0,0,1], [0,1,1], '--', linewidth=2)
  # a, = ax.plot(*zip(*rocs), linewidth=2)
  # a.set_label("UNET")
  # b.set_label("Nasumično gađanje")
  # c.set_label("Idealan slučaj")
  # ax.legend(prop={'size': 18})
  # plt.show()




  # import scipy.misc


  # for i in range(40):
  #   num = len(os.listdir('pics'))//3
  #   scipy.misc.imsave(os.path.join('pics', str(num)+'example-x.png'), xs[i])
  #   scipy.misc.imsave(os.path.join('pics', str(num)+'example-label.png'), labels[i,:,:,0])
  #   scipy.misc.imsave(os.path.join('pics', str(num)+'example-predicts.png'), (outs[i,:,:,0]>=0.64).astype(labels.dtype))

  # plt.imshow(xs[r])
  # plt.show()
  # plt.imshow(labels[r,:,:,0])
  # plt.show()
  # plt.imshow(outs[r,:,:,0])
  # plt.show()