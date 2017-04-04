#!/usr/bin/env python


# Various utility funcitons and classes. Has our input mapping via pygame.
# Also takes screenshots if on Windows or Mac. Pil.ImageGrab does not work on
# Linux. For the actual playing of the game, see play.py, which will do
# screenshots a little different since it will run on Linux.
import sys
import array
import pygame
import wx
from PIL import ImageGrab
from PIL import Image
wx.App()

import numpy as np

from PIL import Image

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def take_screenshot():

    pilImage = ImageGrab.grab()
    img = wx.Image(pilImage.size[0], pilImage.size[1], pilImage.convert("RGB").tobytes())

    # Turning image into smaller one, too big for training on.
    # Size values based on TensorKart, see if it is large enough.
    size = 480, 615

    img.Rescale(615, 480)

    bmp = img.ConvertToBitmap()
    return bmp



def prepare_image(img):
    if(type(img) == wx._core.Bitmap):
        img.CopyToBuffer(Screenshot.image_array)
        img = np.frombuffer(Screenshot.image_array, dtype=np.uint8)

    img = img.reshape(Screenshot.SRC_H, Screenshot.SRC_W, Screenshot.SRC_D)

    im = Image.fromarray(img)
    im = im.resize((Screenshot.IMG_W, Screenshot.IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((Screenshot.IMG_H, Screenshot.IMG_W, Screenshot.IMG_D))

    return img_as_float(im_arr)


class Screenshot:
    SRC_W = 615
    SRC_H = 480
    SRC_D = 3

    OFFSET_X = 0
    OFFSET_Y = 0

    IMG_W = 200
    IMG_H = 66
    IMG_D = 3

    image_array = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));


class XboxController:
    def __init__(self):
        try:
            pygame.init()
            self.joystick = pygame.joystick.Joystick(1)
            self.joystick.init()
        except:
            print 'unable to connect to Xbox Controller'


    def read(self):
        pygame.event.pump()
        x = self.joystick.get_axis(0)
        y = self.joystick.get_axis(1)

        b = self.joystick.get_button(0) # b on gamecube, x on PS4
        a = self.joystick.get_button(1) # a on gamecube, circle on PS4
        xB = self.joystick.get_button(2) # X, Square
        yB = self.joystick.get_button(3) # Y, Triangle
        z = self.joystick.get_button(4) # Z, LB
        rT = self.joystick.get_button(5) # Right trigger, dolphin uses it on RB


        # The rest of the buttons aren't needed, as d-pad is not detected
        # by pygame. We will not be able to train it to taunt (unless we force
        # it to do that at the start of a match)


        # Converting pygame joystick values to Dolphin emulator values
        # turn [-1, 1] to [0, 1], center is .5
        x = 0.5 + (x/2.0)
        y = 0.5 + (y/2.0)


        
        return [x, y, b, a, xB, yB, z, rT]


    def manual_override(self):
        pygame.event.pump()
        return self.joystick.get_button(4) == 1


class Data(object):
    def __init__(self):
        self._X = np.load("data/X.npy")
        self._y = np.load("data/y.npy")
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
    return image_files, joystick_values


# training data viewer
def viewer(sample):
    image_files, joystick_values = load_sample(sample)

    plotData = []

    plt.ion()
    plt.figure('viewer', figsize=(16, 6))

    for i in range(len(image_files)):

        # joystick
        print i, " ", joystick_values[i,:]

        # format data
        plotData.append( joystick_values[i,:] )
        if len(plotData) > 30:
            plotData.pop(0)
        x = np.asarray(plotData)

        # image (every 3rd)
        if (i % 3 == 0):
            plt.subplot(121)
            image_file = image_files[i]
            img = mpimg.imread(image_file)
            plt.imshow(img)

        # plot
        plt.subplot(122)
        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        plt.hold(True)
        plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.draw()
        plt.hold(False)

        plt.pause(0.0001) # seconds
        i += 1


# prepare training data
def prepare(samples):
    print "Preparing data"

    X = []
    y = []

    for sample in samples:
        print sample

        # load sample
        image_files, joystick_values = load_sample(sample)

        # add joystick values to y
        y.append(joystick_values)

        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            vec = prepare_image(image)
            X.append(vec)

    print "Saving to file..."
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/X", X)
    np.save("data/y", y)

    print "Done!"
    return


if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare':
        prepare(sys.argv[2:])
