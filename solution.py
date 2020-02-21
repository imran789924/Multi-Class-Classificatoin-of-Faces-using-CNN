#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:46:56 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import MaxPool2D


classifier = Sequential();

classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Convolution2D(128,3,3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(output_dim=6, activation='relu'))

classifier.add(Dense(output_dim=3, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Dataset/trainig_data',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('Dataset/test_data',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=12,
                    epochs=15,
                    validation_data=test_set,
                    validation_steps=9)

classifier.predict()
