from breakout.shared.core import GameObject
from breakout.config.settings import *


class Pad(GameObject):
    def __init__(self, x, y, image, game, speed=DEFAULT_PAD_SPEED):
        super().__init__(x, y, image, game)
        self.speed = speed
        self.screen_size = self.screen.get_size()

    def collision_detected(self, other_object):
        self.bounce_sound()
        self.game.redraw_pad = True

        # increment game reward for this decision
        half_size_x = self.size_x / 2
        delta_centers = abs((other_object.x + other_object.size_x / 2) - (self.x + self.size_x / 2)) / half_size_x
        #self.game.reward = 20 + 10 * half_size_x / (half_size_x + delta_centers)

        if other_object.x <= self.x < other_object.x + other_object.size_x:
            self.x = other_object.x + other_object.size_x
        elif other_object.x <= self.x + self.size_x < other_object.x + other_object.size_x:
            self.x = other_object.x - self.size_x - 1

    def bounce_sound(self):
        pass

    def set_pad_speed(self, speed):
        self.speed = speed

    def move(self, direction):
        # save old coords
        self.old_x = self.x

        # pad moves only horizontally
        new_x = self.x + direction * self.speed

        # check if it touches walls
        if new_x <= self.speed and direction == -1:
            self.x = 0
        elif new_x + self.size_x >= self.screen_size[0]:
            self.x = self.screen_size[0] - self.size_x - 1
        else:
            self.x = new_x
