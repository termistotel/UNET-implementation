import os
import json
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

  trainDB = trainDB.shuffle(100).repeat().batch(1)
  devDB = devDB.shuffle(100).repeat().batch(1)
  testDB = testDB.shuffle(100).repeat().batch(1)

  iter = testDB.make_one_shot_iterator()

  with tf.Session() as sess:
    im, lab = sess.run(iter.get_next())

    shp = lab.shape[1:3]
    x, y = tuple(shp)
    shp = (x, 2*y)

    tmp = np.zeros(shape = shp + (3,))
    tmp[:, :y, :] = im[0]
    tmp[:, y:, 0] = lab[0, :, :, 0]
    tmp[:, y:, 1] = lab[0, :, :, 0]
    tmp[:, y:, 2] = lab[0, :, :, 0]

    plt.imshow(tmp)
    plt.show()
