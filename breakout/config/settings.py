import os


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

IMAGES_DIR = os.path.join(ASSETS_DIR, 'images')

BRICKS_DIR = os.path.join(IMAGES_DIR, 'bricks')

MUSIC_DIR = os.path.join(ASSETS_DIR, 'music')

SOUNDS_DIR = os.path.join(ASSETS_DIR, 'sounds')

LEVELS_DIR = os.path.join(ROOT_DIR, 'levels')

MATRIX_SIZE_X = 20

MATRIX_SIZE_Y = 20

BACKGROUND_COLOR = (0, 30, 0)

DEFAULT_PAD_SPEED = 6

DEFAULT_BALL_SPEED = 3

PAD_FIXED_HEIGHT = 30

LEFT_MARGIN = 0

TOP_MARGIN = 100

GAME_FPS = 150

TEXT_TO_BRICK_TYPE = {
    'red': 1,
    'blue': 2,
    'green': 3,
    'orange': 4,
    'gray': 5,
    'purple': 6
}

BRICK_TYPE_TO_LOCATION = {
    1: os.path.join(BRICKS_DIR, 'red_brick.png'),
    2: os.path.join(BRICKS_DIR, 'blue_brick.png'),
    3: os.path.join(BRICKS_DIR, 'green_brick.png'),
    4: os.path.join(BRICKS_DIR, 'orange_brick.png'),
    5: os.path.join(BRICKS_DIR, 'gray_brick.png'),
    6: os.path.join(BRICKS_DIR, 'purple_brick.png')
}

BRICK_SCORE = {
    1: 1,  # red brick
    2: 1,  # blue brick
    3: 1,  # green brick
    4: 1,  # orange brick
    5: 1,  # gray brick
    6: 1,  # purple brick
}
