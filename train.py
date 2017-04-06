#!/usr/bin/env python

# This module creates and trains the model. It needs to run in Python 3.5
# py -3.5 works fine as a command so far.


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Lambda
from keras.models import load_model
from keras import backend as K
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float
from PIL import Image

import glob
import random

# Using these static values should cut down on memory needed for each image.
SRC_H = 480
SRC_W = 615
SRC_D = 3

IMG_W = 200
IMG_H = 66
IMG_D = 3

# Loading the samples from the csv.
def load_sample(path):
    print(path)
    images = np.loadtxt(path + "/data.csv", delimiter=',', dtype=str, usecols=(0,))
    joy_vals = np.loadtxt(path + "/data.csv", delimiter=',', usecols=(1,2,3,4,5,6,7,8))
    return images, joy_vals

def prepare_image(img):
    img = img.reshape(SRC_H, SRC_W, SRC_D)

    im = Image.fromarray(img)
    im = im.resize((IMG_W, IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D))


    return img_as_float(im_arr)

# Setting up the model - Need to provide a shape and such to Keras.

model = Sequential()
model.add(Dense(64, input_shape=(66, 200, 3), activation='relu'))
model.add(Dense(64))
model.add(Flatten())
model.add(Dense(8))



model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

# Getting data together for training
# Data is passed in images (viewed as array), labeled by arrays [x, y, b, a, xB, yB, z, rT]


# To make it easy, make a list of the data, then turn it into np.arrays as needed for Keras.
# Keras: input can be a list of np arrays (which is why image is appened as np.array later)

inputX = []; # Inputs are a list of numpy arrays. Append a numpy array for each image
labels = []; # Array of arrays, one array for each image.

# testing the model
testInput = [];
testLabels = [];

# Full and proper training will go here.
# Train on 16 out of 20 matches. Randomly choose which ones are the 4 tests.

path = "samples\\*" #path is relative. samples\1, samples\2, etc. for each recording session (one match)

test = 0

testImage = []
samples = glob.glob(path)
for sample in samples:
    image_files, joy_vals = load_sample(sample)
    # Since glob returns in arbitrary order, assume random enough for testing purposes.
    # Since our test is seeing how it plays, not sure if keeping some data for testing is worth it.
    # Also, not sure how accuracy is measured when we need to interpret our output somewhat to get it to match
    # our predicted (.5 means I pressed button, etc.)
    '''if testCount <= 3:
        for image in image_files:
            image = image[2:-1]
            image = imread(image) # Read image as array
            vec = prepare_image(image)
            testInput.append(vec)
        testLabels.append(joy_vals) # Append labels
        testCount = testCount + 1
    else:'''

    for image in image_files:
        image = image[2:-1]
        image = imread(image) # Read image as array
        vec = prepare_image(image)
        inputX.append(vec)
    labels.append(joy_vals) # Append labels

    inputX = np.asarray(inputX)
    model.fit(inputX, labels, epochs=20)
    if test == 0:
        testImage.append(inputX[0])
    inputX = []
    labels = []
    test = 1
    # Callin fit on a Keras model starts from where it left off. Because there are issues
    # trying to train on all the data at once with passing of the labels, will just train at the end of each sample loop.

#out = model.evaluate(testInput, testLabels)
#print(out)

# testing the save and load of a model.
model.save('test_model100.h5')

# This is how we use the model to play the game: Predict
# Pass an image prepared as above to the predict function.
output = model.predict(np.asarray(testImage))
print(output)

# After Training on 5+ epochs, out output actually makes some sense. 8 floats mostly within range.
# the axis joysticks we will treat as is (negative being 0, beyond 1 being 1)
# The button presses will be a "threshold" type deal, confidence in button type thing. If (.5) or greater, press button
# Might need to fine tune that.


