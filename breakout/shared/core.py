from breakout.config.settings import *


class GameObject:
    def __init__(self, x, y, image, game):
        self.x = x
        self.old_x = x
        self.y = y
        self.old_y = y
        self.initial_draw = True
        self.image = image
        self.game = game
        self.screen = game.screen
        self.size_x, self.size_y = image.get_size()

    def __collides_x(self, other_object):
        return other_object.x < self.x < other_object.x + other_object.size_x or \
               other_object.x < self.x + self.size_x < other_object.x + other_object.size_x

    def __collides_y(self, other_object):
        return other_object.y < self.y < other_object.y + other_object.size_y or \
               other_object.y < self.y + self.size_y < other_object.y + other_object.size_y

    def collides(self, other_object):
        flag = self.would_collide(other_object)
        if flag:
            self.collision_detected(other_object)
            other_object.collision_detected(self)
        return flag

    def would_collide(self, other_object):
        return self.__collides_x(other_object) and self.__collides_y(other_object)

    def collision_detected(self, other_object):
        raise NotImplementedError("Collision method must be implemented for each game object")

    def move(self, *args, **kwargs):
        raise NotImplementedError("Each object must implement move method")

    def draw(self, forced=False):
        # don't draw anything if in training mode
        if not self.game.training_mode:
            # only draw if it changed position or it's the first draw
            if forced or self.old_x != self.x or self.old_y != self.y or self.initial_draw:
                self.screen.fill(BACKGROUND_COLOR, (self.old_x, self.old_y, self.size_x, self.size_y))
                self.screen.blit(self.image, (self.x, self.y))
                self.initial_draw = False

    def remove(self):
        self.screen.fill(BACKGROUND_COLOR, (self.x, self.y, self.size_x, self.size_y))
