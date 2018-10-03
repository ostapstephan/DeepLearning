#!/bin/python3.5
# Ostap Voynarovskiy
# CGML HW4
# October 4 2018
# Professo Curro
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, AveragePooling2D, Input
from keras import regularizers
from keras.optimizers import Adam
#from keras import backend as K

num_classes=10
BATCH_SIZE = 32
epochs = 128
DROP_RATE =.3
weight_decay = 1e-4
#https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
def genTrainAndVal(f,l): #split the features and labels of the training data 80:20 train and validation
	lx=f.shape[0]
	z = f.shape[0]	
	s = np.arange(z)
	np.random.shuffle(s)
	fs = f[s]			# features shuffled
	ls = l[s]			# labels shuffled 
	lx = f.shape[0] 	# len of the features
	nv = int( lx *.2) 	# num validation samp 
	print (fs[nv:].shape, ls[nv:].shape, fs[:nv].shape, ls[:nv].shape)
	return fs[nv:], ls[nv:], fs[:nv], ls[:nv]



def lr_schedule(epoch): #pulled from https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 80:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


# load cifar 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# load cifar 100
#from keras.datasets import cifar100
#(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

#fonvert to float and normalize
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
x_train,x_test = x_train/255,x_test/255

x_t,y_t,x_v,y_v =genTrainAndVal(x_train,y_train)

#print Shapes
print("Training features shape: ", x_t.shape)
print("Validation features shape: ",x_v.shape)
print("Test features shape: ", x_test.shape)

#one hot encode the labels
y_t = keras.utils.to_categorical(y_t,num_classes)
y_v = keras.utils.to_categorical(y_v,num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)
def res_layer(inputs, num_filters=16, kernel_size=3, 
	strides=1, activation='relu', batch_normalization=True, 
	conv_first=True):
	"""2D Convolution-Batch Normalization-Activation stack builder
	# Arguments
		inputs (tensor): input tensor from input image or previous layer
		num_filters (int): Conv2D number of filters
		kernel_size (int): Conv2D square kernel dimensions
		strides (int): Conv2D square stride dimensions
		activation (string): activation name
		batch_normalization (bool): whether to include batch normalization
		conv_first (bool): conv-bn-activation (True) or
			bn-activation-conv (False)
	# Returns
		x (tensor): tensor as input to the next layer
	"""
	conv = Conv2D(num_filters,
		kernel_size=kernel_size,
		strides=strides,
		padding='same',
		kernel_initializer='glorot_uniform',
		kernel_regularizer=l2(weight_decay))

	x = inputs
	if conv_first:
		x = conv(x)
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
	else:
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		x = conv(x)
	return x


'''
model = Sequential()
model.add(Conv2D(32,(4,4),padding='same',kernel_regularizer=regularizers.l2(weight_decay), data_format='channels_last', kernel_initializer='glorot_uniform', input_shape=x_t[0].shape))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(32,(2,2),strides=1,padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(Conv2D(32,(3,3),strides=1,padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(DROP_RATE))

model.add(Conv2D(64,kernel_size=(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay),data_format='channels_last',kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(5,5),padding='same',kernel_regularizer=regularizers.l2(weight_decay),data_format='channels_last',kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(4,4),padding='same',kernel_regularizer=regularizers.l2(weight_decay),data_format='channels_last',kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(DROP_RATE))

model.add(Conv2D(128,kernel_size=(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay),data_format='channels_last',kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay),data_format='channels_last',kernel_initializer='glorot_uniform'))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(num_classes,activation="softmax"))
model.summary()
'''

model.compile(loss="categorical_crossentropy",
	optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
	metrics=['accuracy'])

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_type = 'ResNet%dv%d' % (3, 2)
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [checkpoint,lr_schedule]

datagen = ImageDataGenerator(
	# set input mean to 0 over the dataset
	featurewise_center=False,
	# set each sample mean to 0
	samplewise_center=False,
	# divide inputs by std of dataset
	featurewise_std_normalization=False,
	# divide each input by its std
	samplewise_std_normalization=False,
	# apply ZCA whitening
	zca_whitening=False,
	# epsilon for ZCA whitening
	zca_epsilon=1e-06,
	# randomly rotate images in the range (deg 0 to 180)
	rotation_range=0,
	# randomly shift images horizontally
	width_shift_range=0.1,
	# randomly shift images vertically
	height_shift_range=0.1,
	# set range for random shear
	shear_range=0.,
	# set range for random zoom
	zoom_range=0.,
	# set range for random channel shifts
	channel_shift_range=0.,
	# set mode for filling points outside the input boundaries
	fill_mode='nearest',
	# value used for fill_mode = "constant"
	cval=0.,
	# randomly flip images
	horizontal_flip=True,
	# randomly flip images
	vertical_flip=False,
	# set rescaling factor (applied before any other transformation)
	rescale=None,
	# set function that will be applied on each input
	preprocessing_function=None,
	# image data format, either "channels_first" or "channels_last"
	data_format=None,
	# fraction of images reserved for validation (strictly between 0 and 1)
	validation_split=0.0)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_t)
'''
# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_t, y_t, batch_size=BATCH_SIZE),
		validation_data=(x_v, y_v), steps_per_epoch=x_train.shape[0]/BATCH_SIZE,
		epochs=epochs, verbose=1, workers=4, callbacks=callbacks)
'''
model.fit(datagen.flow(x_t, y_t, batch_size=BATCH_SIZE),
	validation_data =(x_v,y_v),
	steps_per_epoch=x_t.shape[0]/BATCH_SIZE,
	epochs = epochs,
	verbose = 1,
	callbacks=callbacks)


score = model.evaluate(x_test,y_test,verbose=1)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
