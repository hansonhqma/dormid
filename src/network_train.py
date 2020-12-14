#This trains the  network and outputs a HDF5 file of all network weights

import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda, Input
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.initializers import Initializer
from keras import backend as K
from keras.utils import plot_model
import time
import h5py

#Hyperparameters

vnumber = input("Version Number: ")
iter = 1000
batch_size = 16



target = np.load('5kTARGET.npy')


def getBatch(batchsize):
    left = np.load('5kLEFT.npy')
    right = np.load('5kRIGHT.npy')
    batchindex = np.random.randint(0,int(2943-batchsize))
    ubatchindex = np.random.randint(2944,int(4980-batchsize))
    batchleft = np.empty((0,125,100,1), dtype = 'float32')
    batchright = np.empty((0,125,100,1), dtype = 'float32')

    batchtarget = np.concatenate([np.ones(int(batchsize)),np.zeros(int(batchsize))])

    for i in range(batchindex, int(batchindex+batchsize)):
        batchleft = np.concatenate([batchleft, left[i,:,:,:].reshape(1,125,100,1)])
        batchright = np.concatenate([batchright, right[i,:,:,:].reshape(1,125,100,1)])
    for i in range(ubatchindex, int(ubatchindex+batchsize)):
        batchleft = np.concatenate([batchleft, left[i,:,:,:].reshape(1,125,100,1)])
        batchright = np.concatenate([batchright, right[i,:,:,:].reshape(1,125,100,1)])

    return batchleft, batchright, batchtarget

def genSiamese(inputshape):

    inputleft = Input(inputshape)
    inputright = Input(inputshape)

    #Convolutional Network

    model = Sequential()
    model.add(Conv2D(64,(10,10),activation='relu', input_shape = inputshape, kernel_regularizer = l2(2e-4)))
    model.add(MaxPooling2D())

    model.add(Conv2D(128,(7,7),activation='relu', input_shape = inputshape, kernel_regularizer = l2(2e-4)))
    model.add(MaxPooling2D())

    model.add(Conv2D(128,(3,3),activation='relu', input_shape = inputshape, kernel_regularizer = l2(2e-4)))
    model.add(MaxPooling2D())

    model.add(Conv2D(256,(3,3),activation='relu', input_shape = inputshape, kernel_regularizer = l2(2e-4)))

    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    encodeleft = model(inputleft)
    encoderight = model(inputright)

    #Computing absolute difference between encodings

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encodeleft, encoderight])

    prediction = Dense(1, activation='sigmoid')(L1_distance)

    siamese_net = Model(inputs=[inputleft,inputright], outputs = prediction)

    return siamese_net

siamesenet = genSiamese((125,100,1))
siamesenet.summary()

traininghist = np.empty([0])
plotx = np.empty([0])

siamesenet.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.00006))

print("Starting training process")
print("-----------------------------------------------")
for i in range(1,iter+1):
    plotx = np.concatenate([plotx, [i]])
    left, right, target = getBatch(batch_size)
    inputpairs = [left,right]
    loss = siamesenet.train_on_batch(inputpairs, target)
    traininghist = np.concatenate([traininghist, np.array([loss])])
    print("Iteration {} loss: ".format(i), loss)
    decrease = (traininghist[i-2]-loss)/traininghist[i-1]*100
    print("Error decrease: ",decrease,"%")
    print("-------------------------------\n")
    if loss<0.01:
        print(loss, "loss below thresh")
        break

print(traininghist)
plt.scatter(plotx, traininghist)
plt.show()

siamesenet.save('gen{}weights.hdf5'.format(vnumber))
