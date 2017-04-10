# KirbyBot
Training a bot using Tensor Flow (Keras) and training data from us to create a Neural Network to play Smash Bros Melee, as Kirby, on Temple.

## Creators
Zachary Gill

Lucy Holland

Jacob Pankey

## Inspiration and Other Resources

The Idea to use TensorFlow (in this case, Kears) for gameplaying comes from Kevin Hughes and his Tensor Kart project.
[Tensor Kart](http://kevinhughes.ca/blog/tensor-kart)

Other Bots have been made to play Smash. To see what a Bot that wins almost all the time looks like, check out [SmashBot](https://github.com/altf4/SmashBot)

## See it in Action

### 1 Epoch, 20 Samples
[Youtube](https://www.youtube.com/watch?v=w_ZxwWj-GKU&t=1s)

### 1 Epoch, 30 Samples
[Youtube](https://www.youtube.com/watch?v=JfGR3xmGNMk)

### 20 Epochs, 20 Samples
[Youtube](https://www.youtube.com/watch?v=t1zuUD6aLz4)

### 100 Epochs, 30 Samples
[Youtube](https://www.youtube.com/watch?v=dY8kT6irSiw)

## How it works:
KirbyBot is a neural network that has been trained on gameplay data gathered by us (not included in repo to save space). The ANN (Artifical Neural Network) then "plays" Smash Bros on the Dolphin Emulator.

In more detail, the data is gathered with record.py, which takes screenshots of the entire computer screen along with the current state of the controller. The sample rate is about .2 seconds. 

The Controller training data is an array of 2 floats and 6 "bools" (0 or 1). The floats are the X and Y axis values for the main joystick, controlling if we move right, left, up, or down. The bools represent what buttons were pressed. On a gamecube controller, those buttons would be A, B, X, Y, Right Trigger, and Z.

After a recording session is done, we pass the data along to train.py, which generates a model that takes in a 3D Array (Height, Width, RGB Values) and outputs our prediction for button presses (8 floats). The model itself is a multilayer, complicated ANN with at least 64 starting nodes. The output of 8 floats is interpreted as:

 The first 2 floats are as is for joystick values. These are normalized before training to be from 0 to 1, with .5 being the center of the joystick.
 
The next 6 floats are treated as confidence thresholds. If value is above a certain limit (currently, .1), then we consider that button pressed. Otherwise, release. The values tend to be within -.1 and .1, but that could be different based on tuning the model.


To play the game, KirbyBot (with playTest.py), loads a model, then takes a screenshot every .2 seconds (about human visual reaction time). It feeds that image to the model, interprets the output, and pipes the commands to the Dolphin Emulator.

## Compatibility

Piping Commands to Dolphin only works on Unix machines, and our playTest.py has only been tested on Ubuntu.

Record.py uses PIL.Image.ImageGrab, which only works on Windows and Mac. We gathered data and trained on one, played in Linux. 

Python 2.7 to 3.4 for record.py and utils.py (only tested with Python 2.7).

Python 3.5 or greater for train and playTest.py (Keras)


## Requirements

Dolphin Emulator

Controller (XBOX or PS4 interpeted as XBOX recomended.)

The ROM or ISO file of Super Smash Bros. Melee -- NOTE: There can be legal issues surrounding emulating games. Make sure you have the right to emulate the game you are downloading. Generally, if you own the game in an original format of some kind, you are allowed to emulate it.

requirements35.txt has all the package requirements for the playTest.py and Train.py.
requirements27.txt has all the packages for record.py.

## Setup and Running

### Recording
run `pip install -r requirements27.txt` for your version of python (earlier than 3.5)

To run the program, `python record.py` or `py -2.7 record.py`, replace 2.7 with the version you are using.

Some extra work might be required depending on the controller you are using. Pygame will use controller input if the system recongizes it. Dolphin Emulator works great with XInput devices such as XBOX for Windows or PS4 Controllers using [DS4Windows](http://ds4windows.com/). We used PS4, which is why our joystick is number 1 instead of 0 in utils. You might need to change that to 0 if you use an XBOX controller.

We recomend creating a samples folder in the same directory as the record.py, and saving training data in that folder. Record.py will create a sub folder for each recording session.

Make sure your controller is setup for the Dolphin Emulator. See [Dolphin Guide](https://wiki.dolphin-emu.org/index.php?title=Configuring_Controllers#Dolphin_Controller_Configuration)

Once the record button is pressed, a record session starts. It does not stop until the stop button (which has replaced record) is pressed.

Finally, because the whole screen is captured (only the primary monitor is captured if more than 1 is present), make sure the game is full screen and in the front. If there are any images you don't want the model to be trained on, just eliminate that image from the folder and the corresponding entry in the data.csv for that folder.

### Training
run `pip3 install -r requirements35.txt` for Unix, `py -3.5 -m pip install -r requirements35.txt` for Windows

In train.py, make sure to change the code:

Line 108 `model.fit(inputX, labels, epochs=100)` to the number of Epochs (passes over the data) you want. 100 took 24 hours for 30 samples

Line 121 `model.save('test_model100.h5')` with the name you want the model to be saved as. You could enter a path here if you want.

When running train.py, make sure it is in the same directory as the samples folder (unless you change that part of the code as well).

`py -3.5 train.py` for Windows, `python3 train.py` for Unix.

### Playing
Same package installation as for training.

To setup a pipe for the Dolphin Emulator:

`mkdir ~/.dolphin-emu`
`mkdir ~/.dolphin-emu/Pipes`
`mkfifo ~/.dolphin-emu/Pipes/testPipe`

This will make a pipe named testPipe. If you want to rename the pipe, you can, but make sure to rename it in playTest.py as well.

Next, you'll need to open Dolphin Emulator. Find the Controllers option to configure a controller. Choose whatever player you want, and
look for the pipe in the drop down list. Then you have to go through and manually configure each button to be "PRESS [BUTTON]". Finally, make sure the option "Background Input", on the bottom right, is checked. 

Whenever we tried to use pre-made config files for controllers, it did not work, so we assume manual is the only way that works for
KirbyBot.

You can test the pipe with `echo "PRESS A" > ~/.dolphin-emu/Pipes/testPipe`. You should see A being pressed in the controller config panel. It stays pressed until "RELEASE A" is sent.


In the code, make sure to change:

Line 138 `model = load_model("test_model100.h5")` to the name of the trained model you are using.

Line 141 `path = os.path.expanduser('~/.dolphin-emu/Pipes/testPipe') # change if pipe is not this.` if you name the pipe something different.

Use a second controller/player to setup the match. When you are ready to play, start the match and run
`python3 playTest.py`. Make sure the game is full screen and on top of everything else, as the screen capture is similar to record.py

KirbyBot should be playing Smash Bros now!

## Possible Issues
If any packages are missing, manual installation might be needed for each package. wxPython-Phoenix is one of the trouble makers and you may need the wheel, found [here](https://wxpython.org/Phoenix/snapshot-builds/)
