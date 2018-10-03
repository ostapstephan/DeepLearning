# imports
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# define globals
batch_size = 32
num_classes = 10
epochs = 200
imageProcessing = True
normalization = True

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_ver, y_train, y_ver = train_test_split(x_train, y_train, test_size=5000)

if imageProcessing:
  datagen = ImageDataGenerator(
              rotation_range=15,
              width_shift_range=0.1,
              height_shift_range=0.1,
              horizontal_flip=True)

  datagen.fit(x_train)

# properly shape input data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_ver = x_ver.reshape(x_ver.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

# type edits
x_train = x_train.astype('float32')
x_ver = x_ver.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_ver /= 255
x_test /= 255

if normalization:
  for i in range(3):
    x_train[:][:][i] = (x_train[:][:][i] - 0.4914) / 0.247
    x_ver[:][:][i] = (x_ver[:][:][i] - 0.4822) / 0.243
    x_test[:][:][i] = (x_test[:][:][i] - 0.4465) / 0.261

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_ver.shape[0], 'ver samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_ver = keras.utils.to_categorical(y_ver, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# constants
l2weight = 0.0001

# full model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight), input_shape=input_shape))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# build the graph
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.rmsprop(lr=0.001, decay=1e-6), metrics=['accuracy'])

# fit the graph
if imageProcessing:
  model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = 2000, epochs=epochs, verbose=1, validation_data=(x_ver, y_ver))
else:
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_ver, y_ver))

# score the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])