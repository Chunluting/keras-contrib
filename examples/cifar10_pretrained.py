# -*- coding: utf-8 -*-
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.datasets import cifar10
from keras import backend as K

from sklearn.metrics import log_loss

from keras_contrib.applications.resnet_50 import resnet50_model
from keras_contrib.applications.resnet_101 import resnet101_model
from keras_contrib.applications.resnet_152 import resnet152_model
from keras_contrib.applications.densenet_121 import densenet121_model
from keras_contrib.applications.densenet_161 import densenet161_model
from keras_contrib.applications.densenet_169 import densenet169_model

import numpy as np

import sys


def resize(images, factor):
    shape = images.shape
    return np.resize(images, (shape[0], shape[1]*factor, shape[2]*factor, shape[3]))


sys.setrecursionlimit(3000)

# Example to fine-tune on 3000 samples from Cifar10
pretrained_model = 'resnet_50_model'

batch_size = 64
num_classes = 10
epochs = 100

size_factor = 7
img_rows, img_cols = 32 * size_factor, 32 * size_factor
img_channels = 3

# Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0  # 0.0 for data augmentation

# Load our model
if pretrained_model == 'resnet_50_model':
    model = resnet50_model(img_rows, img_cols, img_channels, num_classes)
elif pretrained_model == 'resnet_101_model':
    model = resnet101_model(img_rows, img_cols, img_channels, num_classes)
elif pretrained_model == 'resnet_152_model':
    model = resnet152_model(img_rows, img_cols, img_channels, num_classes)
elif pretrained_model == 'densenet_121_model':
    model = densenet121_model(img_rows, img_cols, img_channels, num_classes)
elif pretrained_model == 'densenet_161_model':
    model = densenet161_model(img_rows, img_cols, img_channels, num_classes)
elif pretrained_model == 'densenet_169_model':
    model = densenet169_model(img_rows, img_cols, img_channels, num_classes)
else:
    raise ValueError('Unsupported model: {}'.format(pretrained_model))

model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
print('Finished compiling')

(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = resize(trainX, size_factor)
testX = resize(trainX, size_factor)
trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX /= 255.
testX /= 255.

Y_train = np_utils.to_categorical(trainY, num_classes)
Y_test = np_utils.to_categorical(testY, num_classes)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32)

generator.fit(trainX, seed=0)

weights_file = pretrained_model + '_CIFAR_10.h5'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')

callbacks = [lr_reducer, early_stopper, model_checkpoint]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size, target_size=(img_rows, img_cols)), steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    verbose=2)

scores = model.evaluate(testX, Y_test, batch_size=batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
