#!/usr/bin/env python

import numpy as np
import os
import shutil
import tkinter as wx
import matplotlib
matplotlib.use('TKAgg')
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigCanvas

from utils import take_screenshot, XboxController

IDLE_SAMPLE_RATE = 1500
SAMPLE_RATE = 200

class MainWindow(wx.Frame):
    """ Main frame of the application
    """
    title = 'Data Acquisition'


    def __init__(self):
        wx.Frame.__init__(self, None)

        # Init controller
        self.controller = XboxController()

         # Create GUI
        self.create_main_panel()

        self.root = wx.Tk()

        # Timer - just use after calls.
        self.root.after(1500, self.on_timer())

        self.recording = False
        self.t = 0


    def create_main_panel(self):
        # Panels
        self.img_panel = wx.Canvas(self)
        self.joy_panel = wx.Canvas(self)
        self.record_panel = wx.Canvas(self)

        # Images
        #img = wx.Image(320,240)
        #self.image_widget = wx.StaticBitmap(self.img_panel, wx.ID_ANY, wx.Bitmap(img))

        # Joystick
        #self.init_plot()
        #self.PlotCanvas = FigCanvas(self.joy_panel, self.fig)

        # Recording
        self.txt_outputDir = wx.Entry(self)
        uid = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.txt_outputDir.insert(0, "samples\\" + uid)

        self.btn_record = wx.Button(self.record_panel, text="Record", width=(20), height=(5))
        self.btn_record.bind("<Button-1>", self.on_btn_record)
        #self.bind(wx.EVT_UPDATE_UI, self.on_update_btn_record, self.btn_record)

        # sizers
        #sizer = wx.BoxSizer(wx.HORIZONTAL)
        #sizer.Add(self.img_panel, 0, wx.ALL, 5)
        #sizer.Add(self.joy_panel, 0, wx.ALL, 5)

        #mainSizer_v = wx.BoxSizer(wx.VERTICAL)
        #mainSizer_v.Add(sizer, 0 , wx.ALL, 5)
        #mainSizer_v.Add(self.record_panel, 0 , wx.ALL, 5)

        # finalize layout
        #self.SetAutoLayout(True)
        #self.SetSizer(mainSizer_v)
        self.pack()


    def init_plot(self):
        self.plotMem = 50 # how much data to keep on the plot
        self.plotData = [[0] * (5)] * self.plotMem # mem storage for plot

        self.fig = Figure((4,3))
        self.axes = self.fig.add_subplot(111)


    def on_timer(self):
        self.poll()
        if self.recording == True:
            self.root.after(200, self.on_timer())
        else:
            self.root.after(1500, self.on_timer())
        # stop drawing if recording to avoid slow downs
        #if self.recording == False:
        #    self.draw()


    def poll(self):
        self.bmp = take_screenshot()
        self.controller_data = self.controller.read()
        self.update_plot()

        if self.recording == True:
            self.save_data()


    def update_plot(self):
        self.plotData.append(self.controller_data) # adds to the end of the list
        self.plotData.pop(0) # remove the first item in the list, ie the oldest


    def save_data(self):
        image_file = self.outputDir+'/'+'img_'+str(self.t)+'.png'
        self.bmp.SaveFile(image_file, wx.BITMAP_TYPE_PNG)

        # make / open outfile
        outfile = open(self.outputDir+'/'+'data.csv', 'a')

        # write line
        outfile.write( image_file + ',' + ','.join(map(str, self.controller_data)) + '\n' )
        outfile.close()

        self.t += 1


    def draw(self):
        # Image
        img = self.bmp.ConvertToImage()
        img = img.Rescale(320,240)
        self.image_widget.SetBitmap( img.ConvertToBitmap() )

        # Joystick
        x = np.asarray(self.plotData)
        self.axes.plot(range(0,self.plotMem), x[:,0], 'r')
        self.axes.hold(True)
        self.axes.plot(range(0,self.plotMem), x[:,1], 'b')
        self.axes.plot(range(0,self.plotMem), x[:,2], 'g')
        self.axes.plot(range(0,self.plotMem), x[:,3], 'k')
        self.axes.plot(range(0,self.plotMem), x[:,4], 'y')
        self.axes.hold(False)
        self.PlotCanvas.draw()


    def on_update_btn_record(self, event):
        label = "Stop" if self.recording else "Record"
        self.btn_record.SetLabel(label)


    def on_btn_record(self, event):
        # pause timer
        self.timer.Stop()

        # switch state
        self.recording = not self.recording

        if self.recording:
            self.start_recording()

        # un pause timer
        if self.recording:
            self.timer.Start(self.rate)
        else:
            self.timer.Start(self.idle_rate)


    def start_recording(self):
        # check that a dir has been specified
        if self.txt_outputDir.IsEmpty():

            msg = wx.MessageDialog(self, 'Specify the Output Directory', 'Error', wx.OK | wx.ICON_ERROR)
            msg.ShowModal() == wx.ID_YES
            msg.Destroy()

            self.recording = False

        else: # a directory was specified
            self.outputDir = self.txt_outputDir.GetValue()
            self.t = 0

            # check if path exists - ie may be saving over data
            if os.path.exists(self.outputDir):

                msg = wx.MessageDialog(self, 'Output Directory Exists - Overwrite Data?', 'Yes or No', wx.YES_NO | wx.ICON_QUESTION)
                result = msg.ShowModal() == wx.ID_YES
                msg.Destroy()

                # overwrite the data
                if result == True:

                    # delete the dir
                    shutil.rmtree(self.outputDir)

                    # re-make dir
                    os.mkdir(self.outputDir)

                # do not overwrite the data
                else: # result == False
                    self.recording = False
                    self.txt_outputDir.SetFocus()

            # no directory so make one
            else:
                os.mkdir(self.outputDir)


    def on_exit(self, event):
        self.Destroy()


if __name__ == '__main__':
    app = MainWindow()
    app.master.title('Test')
    app.mainloop()
