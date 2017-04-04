#!/usr/bin/env python


# This module is used for playing the game. It loads the model,
# takes screenshots at an interval like record.py, passes those images to
# the model, and then interprets the output from the model into commands
# for the Dolphin emulator. It then pipes those commands to the emulator.

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

from PyQt5 import QtWidgets
from PyQt5.QtCore import QBuffer, QIODevice, QByteArray, QSize, Qt
from PyQt5 import QtGui

import os

from io import BytesIO
import sys

import sched,time

# Using these static values should cut down on memory needed for each image.
SRC_H = 480
SRC_W = 615
SRC_D = 3

IMG_W = 200
IMG_H = 66
IMG_D = 3

SAMPLE_RATE = .2

def on_timer(sc):
    image = []
    image.append(np.asarray(screenshot()))

    # Give image to model
    out = model.predict(np.asarray(image))

    # Print output/Pipe to dolphin
    out_to_dolphin(out[0]) # Send the first entry (should be only one) in array
    s.enter(SAMPLE_RATE, 1, on_timer, (sc,))

def int_to_button(i):
    if i == 2:
        return "B"
    elif i == 3:
        return "A"
    elif i == 4:
        return "X"
    elif i == 5:
        return "Y"
    elif i == 6:
        return "Z"
    elif i == 7:
        return "R"

def out_to_dolphin(model_out):

    # Output is: X axis, Y axis, b, a, xB, yB, z, rT
    # Since complex pipe input doesn't seem to be a thing, pipe input after
    # converting it to dolphin readable format.




    output = "SET MAIN "
    output = output + str(model_out[0]) + " " + str(model_out[1]) + "\n" # SET MAIN [X] [Y]

    # write to the pipe
    pipe.write(output)

    i = 2
    while i < 8:
        if float(model_out[i]) < .1:
            output = "RELEASE " + int_to_button(i) + "\n"
        else:
            output = "PRESS " + int_to_button(i) + "\n"
        pipe.write(output)
        i = i+1



def prepare_image(img):
    img = img.reshape(SRC_H, SRC_W, SRC_D)

    im = Image.fromarray(img)
    im = im.resize((IMG_W, IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D))


    return img_as_float(im_arr)


# Get screenshot
def screenshot():
    app = QtWidgets.QApplication(sys.argv)
    
    image = QtWidgets.QApplication.screens()[0].grabWindow(QtWidgets.QApplication.desktop().winId(),0,0,
    QtWidgets.QApplication.desktop().screenGeometry().width(),
    QtWidgets.QApplication.desktop().screenGeometry().height());



    size = QSize(SRC_W, SRC_H)

    image = image.scaled(size)

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly);
    image.save(buffer, "PNG")


    strio = BytesIO()
    strio.write(buffer.data())
    buffer.close()
    strio.seek(0)
    img = imread(strio)

    test = prepare_image(img)
    return test


model = load_model("test_model20.h5")

# open the pipe for writing
path = os.path.expanduser('~/.dolphin-emu/Pipes/testPipe') # change if pipe is not this.
pipe = open(path, 'w', buffering=1)


s = sched.scheduler(time.time, time.sleep)
s.enter(SAMPLE_RATE, 1, on_timer, (s,))
s.run()

