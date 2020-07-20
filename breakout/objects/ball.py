from math import pi, cos, sin

from breakout.shared.core import GameObject
from breakout.shared.utils import get_random_angle
from breakout.objects.pad import Pad


class Ball(GameObject):
    def __init__(self, x, y, image, game, around_angle=-pi / 2):
        super().__init__(x, y, image, game)
        self.angle = get_random_angle(around_angle)
        self.speed = game.ball_speed
        self.screen_size = self.screen.get_size()

    def regularise_angle(self):
        while self.angle >= 2 * pi:
            self.angle -= 2 * pi
        while self.angle < 0:
            self.angle += 2 * pi

    def collision_detected(self, other_object):
        # set new ball position and direction after collision
        temp_old_x = self.x
        temp_old_y = self.y

        # ball collided from bottom
        if (other_object.x <= self.old_x <= other_object.x + other_object.size_x or
                other_object.x <= self.old_x + self.size_x <= other_object.x + other_object.size_x) and \
                self.old_y >= other_object.y + other_object.size_y:
            self.x += cos(self.angle) * self.speed
            self.y = other_object.y + other_object.size_y
            self.angle = 2 * pi - self.angle
        # ball collided from top
        elif (other_object.x <= self.old_x <= other_object.x + other_object.size_x or
                other_object.x <= self.old_x + self.size_x <= other_object.x + other_object.size_x) and \
                self.old_y + self.size_y <= other_object.y:
            self.x += cos(self.angle) * self.speed
            self.y = other_object.y - self.size_y
            self.angle = 2 * pi - self.angle
        # ball collided from right
        elif (other_object.y <= self.old_y <= other_object.y + other_object.size_y or
                other_object.y <= self.old_y + self.size_y <= other_object.y + other_object.size_y) and \
                self.old_x >= other_object.x + other_object.size_x:
            self.x = other_object.x + other_object.size_x
            self.y += sin(self.angle) * self.speed
            self.angle = pi - self.angle
        # ball collided from left
        elif (other_object.y <= self.old_y <= other_object.y + other_object.size_y or
                other_object.y <= self.old_y + self.size_y <= other_object.y + other_object.size_y) and \
                self.old_x + self.size_x <= other_object.x:
            self.x = other_object.x - self.size_x
            self.y += sin(self.angle) * self.speed
            self.angle = pi - self.angle
        # ball collided from corner
        else:
            self.angle += 3 * pi / 4

            # left top corner
            if self.old_x + self.size_x <= other_object.x and self.old_y + self.size_y <= other_object.y:
                self.x = other_object.x - self.size_x
                self.y = other_object.y - self.size_y

            # right top corner
            elif self.old_x >= other_object.x + other_object.size_x and self.old_y + self.size_y <= other_object.y:
                self.x = other_object.x + other_object.size_x
                self.y = other_object.y - self.size_y

            # right bottom corner
            elif self.old_x >= other_object.x + other_object.size_x and \
                    self.old_y >= other_object.y + other_object.size_y:
                self.x = other_object.x + other_object.size_x
                self.y = other_object.y + other_object.size_y

            # left bottom corner
            elif self.old_x + self.size_x <= other_object.x and self.old_y >= other_object.y + other_object.size_y:
                self.x = other_object.x - self.size_x
                self.y = other_object.y + other_object.size_y

        # check if it is the pad we collided with and change angle
        if isinstance(other_object, Pad):
            # how close to the edge did the ball hit
            ball_center = self.x + self.size_x / 2
            hit_percentage = min(abs(ball_center - other_object.x),
                                 abs(ball_center - other_object.x - other_object.size_x))
            hit_percentage = ((other_object.size_x / 2) - hit_percentage) / (other_object.size_x / 2)

            # on which side of the pad (left / right) did the ball hit
            hit_side = -1 if ball_center < other_object.x + other_object.size_x / 2 else 1

            # compute delta angle
            delta_angle = 2 * pi - self.angle if hit_side > 0 else self.angle - pi

            # add a fraction of the missing angle to this ball's angle
            self.angle += delta_angle * 0.8 * hit_side * hit_percentage ** 2

        # save old coords
        self.old_x = temp_old_x
        self.old_y = temp_old_y

    def check_fallen(self):
        # check if the ball has fallen outside of the screen
        return self.y > self.screen_size[1] + self.size_y

    def will_fall(self):
        # check if the ball will fall outside of the screen
        return self.y > self.screen_size[1] - self.size_y

    def move(self):
        # regularise angle to be between 0 and 2pi
        self.regularise_angle()

        # save old coords
        self.old_x = self.x
        self.old_y = self.y

        # compute new coords
        new_x = self.x + cos(self.angle) * self.speed
        new_y = self.y + sin(self.angle) * self.speed
        wall_collision = False

        # check if it bounces of a wall
        # left wall
        if new_x < 0:
            wall_collision = True
            delta_pixels = self.x
            self.x = 0
            self.y += sin(self.angle) * delta_pixels
            self.angle = pi - self.angle
        # top wall
        if new_y < 0:
            wall_collision = True
            delta_pixels = self.y
            self.x += cos(self.angle) * delta_pixels
            self.y = 0
            self.angle = 2 * pi - self.angle
        # right wall
        if new_x + self.size_x >= self.screen_size[0]:
            wall_collision = True
            delta_pixels = self.screen_size[0] - self.x - self.size_x
            self.x = self.screen_size[0] - self.size_x - 1
            self.y += sin(self.angle) * delta_pixels
            self.angle = pi - self.angle

        # no collision - move the ball
        if not wall_collision:
            self.x = new_x
            self.y = new_y
        else:
            self.wall_bounce_sound()

    def wall_bounce_sound(self):
        pass
