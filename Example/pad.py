import enum
import os
import subprocess

@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11

@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1

@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1

class Pad:
    """Writes out controller inputs."""
    def __init__(self, path):
        """Opens the fifo. Blocks until the other end is listening."""
        #self.read, self.write = os.pipe()
        #os.close(self.read)
        #try:
        #    os.mkfifo(path)
        #except OSError:
        #    pass
        #self.write = os.fdopen(fd_write, 'w')
        print(path)

        self.p1 = subprocess.Popen(path, shell=False, stdin=subprocess.PIPE)


    def __del__(self):
        """Closes the fifo."""
        if self.p1:
            self.p1.kill()

    def press_button(self, button):
        """Press a button."""
        assert button in Button
        temp = self.p1.stdin.write(bytes('PRESS {}\n'.format(button.name), 'UTF-8'))
        print(temp)
        self.p1.stdin.flush()

    def release_button(self, button):
        """Release a button."""
        assert button in Button
        self.p1.stdin.write(bytes('RELEASE {}\n'.format(button.name), 'UTF-8'))

    def press_trigger(self, trigger, amount):
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger
        assert 0 <= amount <= 1
        self.p1.stdin.write(bytes('SET {} {:.2f}\n'.format(trigger.name, amount), 'UTF-8'))

    def tilt_stick(self, stick, x, y):
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        try:
          assert 0 <= x <= 1 and 0 <= y <= 1
        except AssertionError:
          import ipdb; ipdb.set_trace()
        self.p1.stdin.write(bytes('SET {} {:.2f} {:.2f}\n'.format(stick.name, x, y), 'UTF-8'))
    
    def send_controller(self, controller):
        for button in Button:
            field = 'button_' + button.name
            if hasattr(controller, field):
                if getattr(controller, field):
                    self.press_button(button)
                else:
                    self.release_button(button)
        
        for trigger in Trigger:
            field = 'trigger_' + trigger.name
            self.press_trigger(trigger, getattr(controller, field))
        
        for stick in Stick:
            field = 'stick_' + stick.name
            value = getattr(controller, field)
            self.tilt_stick(stick, value.x, value.y)
