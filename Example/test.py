import pygame

pygame.init()
print ("Joystics: ", pygame.joystick.get_count())
my_joystick = pygame.joystick.Joystick(0)
my_joystick.init()
clock = pygame.time.Clock()

while 1:
    for event in pygame.event.get():
        print (my_joystick.get_axis(0),  my_joystick.get_axis(1) )
        clock.tick(40)

pygame.quit ()
