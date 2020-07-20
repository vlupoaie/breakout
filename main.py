from DeepQNetwork import DQNAlgorithm
from DeepQNetwork import init_neural_net
from breakout.game.breakout_game import BreakoutGame

import argparse

from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

parser = argparse.ArgumentParser(description='Breakout Game')
parser.add_argument('-m', '--mode', help='Train / Load / Test', required=True)
parser.add_argument('-f', '--file', help='input file', required=False)
args = vars(parser.parse_args())

if args['mode'] == 'Train':
    rn = DQNAlgorithm(input_dim=6, epsilon=1, gamma=0.99, batch_size=20, observe=1000, replay_memory_dim=20000,
                      first_layer_dimension=6, second_layer_dimension=6, filename_to_store_weights="initial_weights")

elif args['mode'] == 'Load':
    rn = DQNAlgorithm(test_flag=2)

elif args['mode'] == 'Test':
    rn = DQNAlgorithm(filename_to_store_weights=args['file'], test_flag=1)

if args['mode'] == 'Test':
    game = BreakoutGame(training_mode=False)

else:
    game = BreakoutGame(training_mode=False)

game.start(rn.game_learner)
