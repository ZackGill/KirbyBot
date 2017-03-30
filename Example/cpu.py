import memory_watcher
import os
import pad
import time
import pygame



class CPU:
    def __init__(self, dump=True, dump_size=3600, dump_dir='experience/'):
        self.dump = dump
        self.dump_size = dump_size
        self.dump_dir = dump_dir

        # TODO This might not always be accurate.
        dolphin_dir = 'C:\\Program Files\\Dolphin\\Dolphin.exe'

        try:
            #print('Creating MemoryWatcher.')
            #self.mw = memory_watcher.MemoryWatcher(dolphin_dir + '/MemoryWatcher/MemoryWatcher')
            print('Creating Pad. Open dolphin now.')
            self.pad = pad.Pad(dolphin_dir)
            self.initialized = True
        except KeyboardInterrupt:
            self.initialized = False

        #self.init_stats(self)
        self.run(self)

    def run(self):
        if not self.initialized:
            return
        print('Starting run loop.')
        try:
            while True:
                self.advance_frame(self)
        except KeyboardInterrupt:
            raise


    def advance_frame(self):
        print("sending things")
        self.pad.press_button(pad.Button.A)


    def init_stats(self):
        self.total_frames = 0
        self.skip_frames = 0
        self.thinking_time = 0

        self.dump_frame = 0
        self.dump_count = 0

    def print_stats(self):
        frac_skipped = self.skip_frames / self.total_frames
        frac_thinking = self.thinking_time * 1000 / self.total_frames
        print('Total Frames:', self.total_frames)
        print('Fraction Skipped: {:.6f}'.format(frac_skipped))
        print('Average Thinking Time (ms): {:.6f}'.format(frac_thinking))
