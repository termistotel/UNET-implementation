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
import scipy.misc

colorDiv = 0.1
batchsize = 2

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

def skip(n, videoIn):
  for i in range(n):
    videoIn.read()

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
  shortDist = np.min(norm, axis=0)<15


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

def edgefun(frame, mask, gmask):
  sobelx = cv2.Sobel(gmask,cv2.CV_64F,1,0,ksize=3)
  sobely = cv2.Sobel(gmask,cv2.CV_64F,0,1,ksize=3)
  edgs = np.array(np.sqrt(np.square(sobelx) + np.square(sobely)))
  edgs[gmask==0] = 0
  edgs = edgs>0

  return np.sum(edgs), edgs


def acquisitionsInit(framecount):
  acquisitions = {
    "površina": {"function": surfun,
              "data": np.zeros(framecount)},
    "opseg": {"function": edgefun,
              "data": np.zeros(framecount)}
  }


  return acquisitions

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("videoFile", metavar="file", type=str, help='Input video file')

  args = p.parse_args()
  inputVideo = args.videoFile

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
  devDB = devDB.shuffle(100).repeat().batch(1)
  testDB = testDB.shuffle(100).repeat().batch(1)

  hparameters = {}
  hparameters = loadHparams("model")

  dirname = "model"

  # Setting up shape info
  shape = testMeta[0]["shape"]
  shp = testMeta[0]["shape"][:2]
  x, y = tuple(shp)
  shp = (x, 2*y)


  # Setting up video input/output
  videoIn = cv2.VideoCapture(inputVideo, apiPreference=cv2.CAP_FFMPEG)
  framecount = videoIn.get(cv2.CAP_PROP_FRAME_COUNT)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  outVideo = cv2.VideoWriter(os.path.join('results', 'demo.avi'),fourcc, 5.0, (2*y, x))
  # skip(150, videoIn)

  # UNET segmentation model initialisation
  model = UNET(trainDB, devDB, testDB, trainMeta, devMeta, testMeta,
           dirname=dirname , hparameters=hparameters)

  # Clustering initalisation
  clust = cluster.DBSCAN(eps=2, min_samples=2)
  average, oldAverage = None, None

  # Values saving
  datas = []

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.saver.restore(sess, os.path.join("model", "model"))

    frameno = 0
    for i in range(150):
    # while True:

      ret, frame = videoIn.read()

      if not ret:
        break

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame, tuple(reversed(shape[:2])))/255
      frame = frame.reshape((1,) + frame.shape)

      # Segment All crystals
      out = sess.run(model.out, feed_dict={model.x: frame})
      mask = out[0,:,:,0]>0.5

      average = segKrist(mask, clust)

      if not average is None:
        if oldAverage is None:
          oldAverage = average

        # Calculate mapping to last frame grains
        average, oldAverage, mapping = rightOrdering(average, oldAverage)

        # print(mapping)

        # Remove duplicates
        mapping1 = []
        maxnum = np.iinfo(mapping.dtype).max
        for m in mapping:
          if m in mapping1:
            mapping1.append(maxnum)
          else:
            mapping1.append(m)

        mapping = mapping1

        # print(mapping)

        for i,m in enumerate(mapping):
          if m < maxnum:
            while (i >= len(datas)):
              datas.append(acquisitionsInit(int(framecount)))
            # Calculate mask for each individual grain
            gmask = np.zeros(shape=mask.shape)
            crystals = np.zeros(shape=(np.sum(mask),))
            crystals[clust.labels_ == m] = 1
            gmask[mask] = crystals

            grainData = datas[i]

            for key in grainData:
              dataType = grainData[key]
              dataType["data"][frameno], _ = dataType["function"](frame, mask, gmask)

      # Preview side by side
      tmp = np.zeros(shape = shp + (3,))
      tmp[:, :y, :] = frame[0]
      tmp[:, y:, 0] = mask
      tmp[:, y:, 1] = mask
      tmp[:, y:, 2] = mask

      # Vrite preview to video
      outVideo.write(np.uint8(tmp*255))

      cv2.imshow('frame', tmp)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      print(frameno)
      frameno+=1


outVideo.release()
videoIn.release()
cv2.destroyAllWindows()


scipy.misc.imsave(os.path.join("results", "lastFrame.png"), frame[0])
scipy.misc.imsave(os.path.join("results", "lastMask.png"), mask.astype(frame.dtype))


realIndex = 0
for i, data in enumerate(datas):
  if data["površina"]["data"][frameno-1] > 0:
    gmask = np.zeros(shape=mask.shape)
    crystals = np.zeros(shape=(np.sum(mask),))
    crystals[clust.labels_ == mapping[i]] = 1
    gmask[mask] = crystals

    print(mapping[i])

    directory = os.path.join("results", str(realIndex))
    if not os.path.exists(directory):
      os.makedirs(directory)

    # cv2.imwrite(os.path.join(directory, "0mask.png"), gmask*255)

    scipy.misc.imsave(os.path.join(directory, "0mask.png"), gmask)
    # plt.imshow(gmask)
    # plt.savefig(os.path.join(directory, "0mask.png"))
    # plt.clf()

    for key in data:
      # cv2.imwrite(os.path.join(directory, key+"-mask.png"), )
      val = data[key]["data"][:frameno]
      # scipy.misc.imsave(os.path.join(directory, key+".png"), val)

      # a, b = data[key]["function"](0, mask, gmask)
      # plt.imshow(b)
      # plt.show()

      fig = plt.figure()
      ax = fig.add_subplot(1,1,1)

      # ax.set_title('ROC krivulja', fontdict={'fontsize': 22})
      ax.set_xlabel('n', fontsize=18)
      ax.set_ylabel(key + ' (broj piksela)', fontsize=20)
      # ax.set_yticks([0.0, 0.5, 1.0])
      # ax.set_yticklabels(["0.0", "0.5", "1.0"])
      # ax.set_xticks(np.linspace(0,1,4))

      a, = ax.plot(val, linewidth=3)
      a.set_label("Izmjereno")
      if key == "opseg":
        val1 = data["površina"]["data"][:frameno]
        pred = lambda P: 6/np.sqrt(np.sqrt(3))*np.sqrt(P)
        b, = ax.plot(pred(val1)*np.sum(val)/np.sum(pred(val1)), linewidth=2)
        b.set_label("Predviđeno")

      # a.set_label("UNET")
      # b.set_label("Nasumično gađanje")
      # c.set_label("Idealan slučaj")
      ax.legend(prop={'size': 20})

      # plt.plot(val)

      for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
      for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

      plt.savefig(os.path.join(directory, key+".png"))
      plt.clf()
      plt.close('all')

      realIndex += 1
