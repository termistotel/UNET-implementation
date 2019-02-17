import numpy as np
import os
import cv2
import random
import json
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

def readImage(file, newsize):
  img = cv2.imread(file)
  img = cv2.resize(img, newsize)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
  return img.astype(np.float32)

def readLabel(file, newsize):
  img = cv2.imread(file, 0)
  img = cv2.resize(img, newsize)
  img = img.reshape(img.shape + (1,))/255.0
  return (img > 0.5).astype(np.float32)

def savedset(name, dirs, files, shape):
  with open(name + ".data", 'wb') as data:
    for imgFile, labelFile in zip(*files):
      img = readImage(os.path.join(dirs[0], imgFile), shape)
      label = readLabel(os.path.join(dirs[1], labelFile), shape)
      data.write((img).reshape(-1).tostring())
      data.write((label).reshape(-1).tostring())
  shape = (img.shape[:3], label.shape[:3])
  length = (len(files[0]), len(files[1]))
  types = (img.dtype, label.dtype)

  with open(name + ".metadata", 'w') as mdata:
    mdata.write(json.dumps({'shape': shape[0], 'length': length[0], "dtype": str(types[0])}))
    mdata.write("\n")
    mdata.write(json.dumps({'shape': shape[1], 'length': length[1], "dtype": str(types[1])}))

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=False, type=str, default="data/", help='input image dir')
  p.add_argument("--labels", required=False, type=str, default="labels/", help='input label dir')
  p.add_argument("--rescale", required=False, type=float, default=1.0, help='resize scale')

  args = p.parse_args()
  datadir = args.data
  labeldir = args.labels
  rescale = args.rescale

  imageList = sorted(os.listdir(datadir))
  sizes = map(lambda file: Image.open(os.path.join(datadir, file)).size, imageList)
  minShape = sorted(list(set(sizes)))[0]
  shape = (int(minShape[0]*rescale), int(minShape[1]*rescale))

  print("Reshaping images to: " + str(shape))
  input("Press any key to continue...")

  # imgs = np.array([readImage(os.path.join(datadir, file), shape) for file in sorted(os.listdir(datadir))], dtype=np.float32)
  # labels = np.array([readLabel(os.path.join(labeldir, file), shape) for file in sorted(os.listdir(labeldir))], dtype=np.float32)
  # imgs = imgs/255
  # labels = labels/255

  # random.seed(1337)
  shuffleList = np.arange(len(imageList))
  np.random.shuffle(shuffleList)
  dataList = np.array(sorted(os.listdir(datadir)))[shuffleList]
  labelList = np.array(sorted(os.listdir(labeldir)))[shuffleList]

  # # Split images into train, dev and test set
  # trainx, devtestx, trainy, devtesty = train_test_split(imgs, labels, test_size=0.2)
  # devx, testx, devy, testy = train_test_split(devtestx, devtesty, test_size=0.5)

  # Split images into train, dev and test set
  trainx, devtestx, trainy, devtesty = train_test_split(dataList, labelList, test_size=0.2)
  devx, testx, devy, testy = train_test_split(devtestx, devtesty, test_size=0.5)

  savedset( "train", (datadir, labeldir), (trainx, trainy), shape )
  print("da")
  savedset( "dev", (datadir, labeldir), (devx, devy), shape )
  print("da")
  savedset( "test", (datadir, labeldir), (testx, testy), shape )
  print("da")


  # Save train set database
  # with open("train.data", 'wb') as data:
  #   for imgFile, labelFile in zip(trainx, trainy):
  #     img = readImage(os.path.join(datadir, imgFile))
  #     label = readImage(os.path.join(labeldir, labelFile))
  #     data.write((img).reshape(-1).tostring())
  #     data.write((label).reshape(-1).tostring())
  #   trainShape = (img.shape, label.shape)
  #   trainLength = (len(img), len(label))
  #   trainTypes = (img.dtype, label.dtype)

  # # Save dev set database
  # with open("dev.data", 'wb') as data:
  #   for imgFile, labelFile in zip(devx, devy):
  #     data.write((img).reshape(-1).tostring())
  #     data.write((label).reshape(-1).tostring())

  # # Save test set database
  # with open("test.data", 'wb') as data:
  #   for img, label in zip(testx, testy):
  #     data.write((img).reshape(-1).tostring())
  #     data.write((label).reshape(-1).tostring())

  # Save train set metadata
  # with open("train.metadata", 'w') as mdata:
  #   mdata.write(json.dumps({'shape': trainx.shape[1:], 'length': len(trainx), "dtype": str(trainx.dtype)}))
  #   mdata.write("\n")
  #   mdata.write(json.dumps({'shape': trainy.shape[1:], 'length': len(trainy), "dtype": str(trainy.dtype)}))

  # # Save train set metadata
  # with open("dev.metadata", 'w') as mdata:
  #   mdata.write(json.dumps({'shape': devx.shape[1:], 'length': len(devx), "dtype": str(devx.dtype)}))
  #   mdata.write("\n")
  #   mdata.write(json.dumps({'shape': devy.shape[1:], 'length': len(devy), "dtype": str(devy.dtype)}))

  # # Save test set metadata
  # with open("test.metadata", 'w') as mdata:
  #   mdata.write(json.dumps({'shape': testx.shape[1:], 'length': len(testx), "dtype": str(testx.dtype)}))
  #   mdata.write("\n")
  #   mdata.write(json.dumps({'shape': testy.shape[1:], 'length': len(testy), "dtype": str(testy.dtype)}))
