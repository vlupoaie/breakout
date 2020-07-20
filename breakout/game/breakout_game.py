import pygame
import platform
from math import cos, sin

from breakout.config.settings import *
from breakout.shared.utils import get_screen_size
from breakout.objects import Ball, Brick, Pad


class BreakoutGame:
    def __init__(self, training_mode=False):
        self.training_mode = training_mode
        self.screen_size = get_screen_size()
        if not self.training_mode or platform.system() == 'Windows':
            self.screen = pygame.display.set_mode(self.screen_size, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.screen_size)
        self.last_loaded = None
        self.last_volume = 0
        self.reward = 0
        if not self.training_mode:
            pygame.init()
            pygame.mixer.init()
            self.clock = pygame.time.Clock()
            pygame.mouse.set_visible(False)
        self.controller = None
        self.lives_remaining = None
        self.ball_speed = None
        self.pad = None
        self.balls = None
        self.bricks = None
        self.score = None
        self.level_matrix = None
        self.redraw_pad = None
        self.redraw_brick = None
        self.pad_collision = None
        self.initialise_game()

    def user_mouse_controller(self, state=None, reward=None):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        threshold = self.pad.speed
        delta_pixels = mouse_x - self.pad.x
        direction = 0
        if abs(delta_pixels) > threshold:
            direction = 1 if delta_pixels > 0 else -1
        return direction

    def initialise_game(self):
        self.redraw_pad = False
        self.redraw_brick = [False for _ in range(MATRIX_SIZE_X * MATRIX_SIZE_Y)]
        self.lives_remaining = 3
        if self.training_mode:
            self.ball_speed = 8 * DEFAULT_BALL_SPEED
        else:
            self.ball_speed = DEFAULT_BALL_SPEED
        self.score = 0
        self.pad = self.get_new_pad(self.screen_size)
        self.balls = [self.get_ball_on_pad()]
        self.pad_collision = [True]
        self.load_level()

    def get_new_pad(self, screen_size):
        pad_image = pygame.image.load(os.path.join(IMAGES_DIR, 'pad_large.png'))
        pad_size = pad_image.get_size()
        pad_x = (screen_size[0] - pad_size[0]) / 2
        if self.training_mode:
            new_pad = Pad(pad_x, screen_size[1] - PAD_FIXED_HEIGHT - pad_size[1],
                          pad_image, self, speed=8 * DEFAULT_PAD_SPEED)
        else:
            new_pad = Pad(pad_x, screen_size[1] - PAD_FIXED_HEIGHT - pad_size[1], pad_image, self)
        return new_pad

    def get_ball_on_pad(self):
        ball_image = pygame.image.load(os.path.join(IMAGES_DIR, 'ball_large.png'))
        ball_size = ball_image.get_size()
        ball_x = (self.pad.size_x - ball_size[0]) / 2 + self.pad.x
        ball_y = self.pad.y - ball_size[1]
        new_ball = Ball(ball_x, ball_y, ball_image, self)
        return new_ball

    def load_level(self, name='hearts'):
        level_file = os.path.join(LEVELS_DIR, '{}.lvl'.format(name))
        if not os.path.isfile(level_file):
            raise FileNotFoundError('Level file "{}" not found'.format(name))

        music_file = os.path.join(MUSIC_DIR, '{}.mp3'.format(name))
        if not os.path.isfile(music_file):
            raise FileNotFoundError('Music file "{}" not found'.format(name))

        # starting music
        if not self.training_mode:
            if self.last_loaded != name:
                pygame.mixer.music.load(music_file)
                pygame.mixer.music.play(loops=-1)
                self.last_loaded = name

        # reading level file and parsing it
        level_matrix = []
        with open(level_file, 'r') as handle:
            content = [line.strip() for line in handle.readlines() if line.strip()]
        if len(content) != MATRIX_SIZE_Y:
            raise Exception('Level file "{}" does not respect format'.format(name))
        for line in content:
            try:
                # temp_line = [TEXT_TO_BRICK_TYPE[item.strip()] for item in line.split() if item.strip()]
                temp_line = [int(item.strip()) for item in line.split() if item.strip()]
            except KeyError:
                raise KeyError('Unknown brick type in level "{}"'.format(name))
            if len(temp_line) != MATRIX_SIZE_X:
                raise Exception('Level file "{}" does not respect format'.format(name))
            level_matrix.append(temp_line)

        # bricks images only load once use more times
        brick_images = {}

        # setting level matrix
        self.level_matrix = level_matrix
        # setting level bricks array
        self.bricks = []
        for count_line, line in enumerate(level_matrix):
            for count_elem, elem in enumerate(line):
                brick_image = brick_images.get(elem)
                if not brick_image:
                    brick_image = pygame.image.load(BRICK_TYPE_TO_LOCATION[elem])
                    brick_images[elem] = brick_image
                image_size_x, image_size_y = brick_image.get_size()
                brick_x = count_elem * (image_size_x + 1) + LEFT_MARGIN
                brick_y = count_line * (image_size_y + 1) + TOP_MARGIN
                new_brick = Brick(brick_x, brick_y, brick_image, self, elem)
                self.bricks.append(new_brick)

        # clean screen from old level
        if not self.training_mode:
            self.screen.fill(BACKGROUND_COLOR)

    def register_controller(self, next_move_function):
        self.controller = next_move_function

    def increment_score(self, value):
        if value > 0:
            self.score += value

    def first_time_draw(self):
        # draw bricks
        for item in self.bricks:
            item.draw()

        # draw pad
        self.pad.draw()

        # draw ball
        for ball in self.balls:
            ball.draw()

    def start(self, controller_function=None):
        if not controller_function:
            self.register_controller(self.user_mouse_controller)
        else:
            self.register_controller(controller_function)

        self.first_time_draw()
        while True:
            if not self.training_mode:
                # set FPS
                self.clock.tick(GAME_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    this_volume = pygame.mixer.music.get_volume()
                    pygame.mixer.music.set_volume(self.last_volume)
                    self.last_volume = this_volume

            # change pad direction
            bricks_left = sum([1 if item else 0 for item in self.bricks])
            lowest_x_delta = None
            lowest_y_delta = None
            lowest_euclidian_distance = None
            ball_angle = None
            ball_x_speed = None
            ball_y_speed = None
            ball_center_x = None
            ball_center_y = None
            # remember deltas for the closest balls only
            for ball_counter, ball in enumerate(self.balls):
                x_delta = ball.x + ball.size_x / 2 - self.pad.x - self.pad.size_x / 2
                y_delta = abs(ball.y + ball.size_y / 2 - self.pad.y - self.pad.size_y / 2)
                euclidian_distance = (x_delta ** 2 + y_delta ** 2) ** 0.5
                if lowest_euclidian_distance is None or euclidian_distance < lowest_euclidian_distance:
                    lowest_euclidian_distance = euclidian_distance
                    lowest_x_delta = x_delta
                    lowest_y_delta = y_delta
                    ball_angle = ball.angle
                    ball_x_speed = cos(ball_angle) * ball.speed
                    ball_y_speed = sin(ball_angle) * ball.speed
                    ball_center_x = int(ball.x + ball.size_x / 2)
                    ball_center_y = int(ball.y + ball.size_y / 2)

            # build state and pass it to controller function
            pad_center_x = int(self.pad.x + self.pad.size_x / 2)
            # state = [bricks_left, lowest_x_delta, lowest_y_delta, lowest_euclidian_distance,
            #          ball_x_speed, ball_y_speed, ball_center_x, ball_center_y, pad_center_x]
            state = [lowest_x_delta, lowest_y_delta, ball_center_x, ball_center_y, pad_center_x, lowest_euclidian_distance]

            # get new pad direction
            pad_direction = self.controller(state, self.reward)
            if self.pad.x <= self.balls[0].x + self.balls[0].size_x/2 <= self.pad.x + self.pad.size_x:
                self.reward = 1
            else:
                self.reward = -1


            #self.reward = -lowest_euclidian_distance / 200

            # move pad and redraw
            self.pad.move(pad_direction)
            self.pad.draw()

            # add objects to test collision with balls
            all_objects = [self.pad]
            for brick in self.bricks:
                all_objects.append(brick)

            # move balls
            for ball in self.balls:
                ball_collided = False
                # collided_pad = False
                for game_object in all_objects:
                    if not game_object:
                        continue
                    if ball.collides(game_object):
                        ball_collided = True
                        # if isinstance(game_object, Pad):
                        #     collided_pad = True
                # self.pad_collision.append(collided_pad)
                if not ball_collided:
                    ball.move()
                ball.draw()

            # check where the ball will fall when it leaves screen
            # for count_collision, collision_flag in enumerate(self.pad_collision):
            #     if collision_flag:
            #         self.simulate_collision_course(self.balls[count_collision])
            # self.pad_collision = []

            # check if any ball fell out of the screen and delete it
            for counter, ball in enumerate(self.balls):
                if ball.check_fallen():
                    del self.balls[counter]

            # if no balls left then restart game
            if not self.balls:
                self.initialise_game()
                self.first_time_draw()
                self.reward = -4000
                continue

            # in case the pad had a collision, redraw it
            self.pad.draw(forced=self.redraw_pad)

            # in case any brick had a collision redraw surrounding ones
            for index, value in enumerate(self.redraw_brick):
                if value:
                    self.redraw_brick[index] = False
                    if self.bricks[index]:
                        self.bricks[index].draw(forced=True)

            if not self.training_mode:
                # redraw screen
                pygame.display.update()

    def simulate_collision_course(self, original_ball):
        this_ball = Ball(original_ball.x, original_ball.y, original_ball.image, self)
        this_ball.angle = original_ball.angle
        this_ball.speed = original_ball.speed * 10

        bricks_copies = []
        for brick in self.bricks:
            if not brick:
                continue
            new_brick = Brick(brick.x, brick.y, brick.image, self, brick.type)
            bricks_copies.append(new_brick)

        while not this_ball.will_fall():
            ball_collided = False
            for count_brick, brick in enumerate(bricks_copies):
                if not brick:
                    continue
                if this_ball.would_collide(brick):
                    ball_collided = True
                    this_ball.collision_detected(brick)
                    bricks_copies[count_brick] = None
            if not ball_collided:
                this_ball.move()

        this_ball.draw()


if __name__ == '__main__':
    game = BreakoutGame()
    game.start()
