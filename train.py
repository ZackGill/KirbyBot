#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float
from PIL import Image


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

# Setting up the model




model = Sequential()
model.add(Dense(64, input_shape=(66, 200, 3)))
model.add(Dense(64))
model.add(Flatten())
model.add(Dense(8))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Getting data together for training
# Data is passed in images (viewed as array), labeled by arrays [x, y, b, a, xB, yB, z, rT]


# To make it easy, make a list of the data, then turn it into np.arrays as needed for Keras.
# Keras: input can be a list of np arrays (which is why image is appened as np.array later)
# Keras: Labels must be numpy arrays, so we convert the list as we pass it in.

inputX = []; # Inputs are a list of numpy arrays. Append a numpy array for each image
labels = []; # Array of arrays, one array for each image.

path = "samples/1" #path is relative. samples\1, samples\2, etc. for each recording session (one match)

# For a quick test, just training on samples\1, testing on samples\2.

# Loads the samples in
images, joy_vals = load_sample(path)

# Appending images to inputX
for image_file in images:
    image_file = image_file[2:-1]
    image = imread(image_file) # Reads image from a file as an array
    vec = prepare_image(image)
    inputX.append(vec)

# Appending joystick inputs to labels

labels = joy_vals

# training the model

inputX = np.asarray(inputX)




model.fit(inputX, labels, epochs=100)

# testing the model
testInput = [];
testLabels = [];

testImages, test_vals = load_sample("samples/2")
for image_file in testImages:
    image_file = image_file[2:-1]
    image = imread(image_file)
    vec = prepare_image(image)
    testInput.append(vec)

testLabels.append(test_vals)

testInput = np.asarray(testInput)





out = model.evaluate(testInput, testLabels)
print(out)
