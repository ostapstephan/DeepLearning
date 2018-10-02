#!/bin/python3.6

#Luka Lipovac - Homework #3
#Machine Learning - Chris Curro

#Credits to treszkai of https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#for saving me


##Things to test!
#
#Amount of layers
#Layer Sizes
#Maxpooling in each layer
#Dropout
#Shuffling in fit
#try rmsprop

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K

EPOCHS = 10
BATCH_SIZE = 32
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = sgd

SIZE_L1 = 96
SIZE_L2 = 96
SIZE_L3 = 96

SIZE_L4 = 192
SIZE_L5 = 192
SIZE_L6 = 192

SIZE_L7 = 192
SIZE_L8 = 192

SIZE_DENSE_1 = 512
SIZE_DENSE_2 = 512
SIZE_OUT = 10

IMAGE_SIZE_X, IMAGE_SIZE_Y = 32, 32
IMAGE_DEPTH = 3 #RGB
VAL_PERC = .05

DROPOUT_1 = 0.5
DROPOUT_2 = 0.5
L2_LAM = 0.001

# the data, split between train and test sets
(images_train, labels_train), (images_test, labels_test) = keras.datasets.cifar10.load_data()
input_shape = images_train.shape[1:]

images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train, images_test = images_train/255.0, images_test/255.0
labels_train, labels_test = keras.utils.to_categorical(labels_train), keras.utils.to_categorical(labels_test)
images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=VAL_PERC)


#Make Layers
model = Sequential()
model.add(Conv2D(SIZE_L1,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(SIZE_L2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(SIZE_L3,
                kernel_size=(3, 3),
                padding='same',
                subsample=(2,2),
                activation='relu'))
model.add(Dropout(DROPOUT_1))

model.add(Conv2D(SIZE_L4,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(SIZE_L5,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(SIZE_L6,
                kernel_size=(3, 3),
                padding='same',
                subsample=(2,2),
                activation='relu'))
model.add(Dropout(DROPOUT_2))

model.add(Conv2D(SIZE_L7,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(SIZE_L8,
                kernel_size=(1, 1),
                padding='valid',
                activation='relu'))
model.add(Conv2D(SIZE_OUT,
                kernel_size=(1, 1),
                padding='valid'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(images_train, labels_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(images_val, labels_val))


score = model.evaluate(images_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
