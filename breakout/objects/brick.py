import pygame

from breakout.shared.core import GameObject
from breakout.config.settings import *


class Brick(GameObject):
    def __init__(self, x, y, image, game, brick_type):
        super().__init__(x, y, image, game)
        self.type = brick_type
        if not self.game.training_mode:
            self.brick_sound = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, 'brick_broken.wav'))

    def collision_detected(self, other_object):
        # break sound
        self.break_sound()

        # increment game score according to this brick type
        self.game.increment_score(BRICK_SCORE[self.type])

        # increment game reward for this decision
        #self.game.reward = 5

        # remove this brick and redraw background
        self.remove()

        # remove this brick from game's bricks array
        for counter, brick in enumerate(self.game.bricks):
            if brick is self:
                # mark brick as broken
                self.game.bricks[counter] = None

                # redraw bricks surrounding this one
                bricks_count = MATRIX_SIZE_X * MATRIX_SIZE_Y
                redraw_indexes = [
                    counter - MATRIX_SIZE_X - 1, counter - MATRIX_SIZE_X, counter - MATRIX_SIZE_X + 1,
                    counter - 1, counter + 1,
                    counter + MATRIX_SIZE_X - 1, counter + MATRIX_SIZE_X, counter + MATRIX_SIZE_X + 1,
                ]
                redraw_indexes = [item for item in redraw_indexes if 0 <= item < bricks_count]
                for index in redraw_indexes:
                    if self.game.bricks[index]:
                        self.game.redraw_brick[index] = True
                break

    def break_sound(self):
        pass

    def move(self):
        # bricks don't move
        return
