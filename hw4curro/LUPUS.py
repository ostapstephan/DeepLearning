#!/bin/python3.6

#Luka Lipovac - Homework #4
#Machine Learning - Chris Curro

#Credits to treszkai of https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/


from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import rmsprop
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 100:
        lrate = 0.0002
    elif epoch > 75:
        lrate = 0.0005       
    return lrate

EPOCHS = 125
BATCH_SIZE = 32
weight_decay = 1e-4
regularizer = l2(weight_decay)
optimizer = rmsprop(lr=0.001,decay=1e-6)

SIZE_L1 = 32
SIZE_L2 = 32
SIZE_L3 = 64
SIZE_L4 = 64
SIZE_L5 = 128
SIZE_L6 = 128

SIZE_DENSE = 1024
SIZE_OUT = 10

IMAGE_SIZE_X, IMAGE_SIZE_Y = 32, 32
IMAGE_DEPTH = 3 #RGB
VAL_PERC = .05

DROPOUT_2 = 0.2
DROPOUT_4 = 0.3
DROPOUT_6 = 0.4

# the data, split between train and test sets
(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()
input_shape = images_train.shape[1:]

images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train, images_test = images_train/255.0, images_test/255.0
labels_train, labels_test = to_categorical(labels_train), to_categorical(labels_test)
images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=VAL_PERC)


#Make Layers
model = Sequential()

model.add(Conv2D(SIZE_L1, (3,3), 
                 padding='same', 
                 activation='elu',
                 kernel_regularizer=regularizer, 
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(SIZE_L2, (3,3), 
                 padding='same',
                 activation='elu',
                 kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(DROPOUT_2))
 
model.add(Conv2D(SIZE_L3, (3,3), 
                 padding='same',
                 activation='elu',
                 kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Conv2D(SIZE_L4, (3,3), 
                 padding='same',
                 activation='elu',
                 kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(DROPOUT_4))
 
model.add(Conv2D(SIZE_L5, (3,3), 
                 padding='same',
                 activation='elu',
                 kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Conv2D(SIZE_L6, (3,3), 
                 padding='same',
                 activation='elu',
                 kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(DROPOUT_6))
 
model.add(Flatten())
model.add(Dense(SIZE_OUT, 
                activation='softmax'))
 

model.compile(loss=categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(images_train, labels_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(images_val, labels_val),
          shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule)])

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

#test
score = model.evaluate(images_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
