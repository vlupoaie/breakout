from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils.visualize_util import plot
from keras.models import model_from_json
import numpy as np
import random
import csv
import ast
import pickle

import matplotlib.pyplot as plt


def init_neural_net(input_dimension, params):
    model = Sequential()

    # first layer
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(input_dimension,)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # output layer
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    return model


def log_results(filename, loss_log):
    with open(filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class DQNAlgorithm:

    def __init__(self, input_dim=9, epsilon=1, gamma=1, batch_size=50, observe=1000, replay_memory_dim=3000,
                 first_layer_dimension=9, second_layer_dimension=9, filename_to_store_weights='initial_weights',
                 counter_observer=0, model=None, replay_memory=list(), test_flag=0):

        self.test_flag = test_flag

        if self.test_flag == 2:  # load

            f = open('backup-data', "rb")
            store = pickle.loads(f.read())
            f.close()

            self.input_dim = store['input_dim']
            self.epsilon = store['epsilon']
            self.gamma = store['gamma']
            self.batch_size = store['batch_size']
            self.observe = store['observe']
            self.replay_memory_dim = store['replay_memory_dim']
            self.list_params_layers = [store['first_layer_dimension'], store['second_layer_dimension']]
            self.filename_to_store_weights = store['filename_to_store_weights']
            self.replay_memory = store['replay_memory']
            self.counter_observer = store['counter_observer']

        elif self.test_flag == 0:  # train

            self.input_dim = input_dim
            self.epsilon = epsilon
            self.gamma = gamma
            self.batch_size = batch_size
            self.observe = observe
            self.replay_memory_dim = replay_memory_dim
            self.list_params_layers = [first_layer_dimension, second_layer_dimension]
            self.filename_to_store_weights = filename_to_store_weights
            self.replay_memory = replay_memory

            self.counter_observer = counter_observer

        else:  # test

            self.observe = 99999999
            self.counter_observer = 0
            self.epsilon = 0

        # load model
        if self.test_flag == 1 or self.test_flag == 2:

            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(filename_to_store_weights)
            print("Loaded model from disk")
            loaded_model.compile(loss='mse', optimizer='rmsprop')
            self.model = loaded_model

        else:
            self.model = init_neural_net(self.input_dim, self.list_params_layers)

        self.counter_over = 0
        self.loss_log = []
        self.store = dict()

        self.old_state = None
        self.new_state = None
        self.action = None
        self.reward = None

    def reset_variables_if_game_over(self):

        self.old_state = None
        self.action = None
        self.reward = None

    def choose_action(self):

        # choose action random if number lower than epsilon or we are not done observing
        if (random.random() < self.epsilon or self.counter_observer < self.observe) and self.test_flag == 0:
            self.action = np.random.randint(0, 3)
            print("--------------Random Action--------------")
        else:

            print("---------------MAX ACTION-----------------")
            # predict Q values for each action
            q_value = self.model.predict(self.new_state, batch_size=1)

            # choose action whose value is max
            self.action = (np.argmax(q_value))
            print("------------ACTION- " + str(self.action))

    def game_learner(self, state, reward, training_frames=10000):

        state = np.array([state])
        print("==================STATE=============")
        print(str(state))
        print("==================REWARD============")
        print(str(reward))

        self.new_state = state

        self.reward = reward

        self.counter_observer += 1

        # if not initial state
        if self.old_state != None and self.test_flag != 1:
            # add to replay memory tuple (s, a, r, s')

            self.replay_memory.append((self.old_state, self.action, self.reward, self.new_state))

            print("Frame: " + str(self.counter_observer) + "------ Reward: " + str(
                self.reward) + "------ Epsilon: " + str(self.epsilon))

            # if the observing it's done, start training
            if self.counter_observer > self.observe:

                # if the replay memory is full, pop the oldest element
                if len(self.replay_memory) > self.replay_memory_dim:
                    self.replay_memory.pop(0)

                # choose random a batch from replay memory
                minibatch = random.sample(self.replay_memory, self.batch_size)

                # prepare states and target values of the Q function for training
                states, target_values = self.process_minibatch(minibatch)

                # train the model on this batch
                history = LossHistory()
                self.model.fit(states, target_values, batch_size=self.batch_size, nb_epoch=10, verbose=0,
                               callbacks=[history])
                self.loss_log.append(history.losses)
                print("---------------Losses: " + str(history.losses))

            # prepare transition to next state
        self.old_state = self.new_state

        # decrement epsilon if it didn't reach 0 yet and if we finished the observation
        if self.epsilon > 0 and self.counter_observer > self.observe:
            self.epsilon -= (1 / training_frames)

        if self.counter_observer % 3000 == 0 and self.test_flag == 0:
            f = open("backup-data", "wb")

            self.store['counter_observer'] = self.counter_observer
            self.store['epsilon'] = self.epsilon
            self.store['new_state'] = self.new_state
            self.store['old_state'] = self.old_state
            self.store['action'] = self.action
            self.store['reward'] = self.reward
            self.store['replay_memory'] = self.replay_memory
            self.store['gamma'] = self.gamma
            self.store['batch_size'] = self.batch_size
            self.store['observe'] = self.observe
            self.store['replay_memory_dim'] = self.replay_memory_dim
            self.store['input_dim'] = self.input_dim
            self.store['first_layer_dimension'] = self.list_params_layers[0]
            self.store['second_layer_dimension'] = self.list_params_layers[1]
            self.store['filename_to_store_weights'] = self.filename_to_store_weights
            f.write(pickle.dumps(self.store))

            print("Saving data...")

        self.choose_action()

        # save trained network every 1 000 frames
        if self.counter_observer % 1000 == 0 and self.test_flag == 0:
            self.model.save_weights(self.filename_to_store_weights + "_" + str(self.counter_observer) + ".h5",
                                    overwrite=True)
            print("Saving model %s - %d" % (self.filename_to_store_weights, self.counter_observer))

            model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)

            plt.plot(self.loss_log)
            plt.title("Loss Graphics")
            plt.savefig("fig.png")

        # action to be -1, 0, 1
        return (self.action - 1)

    def process_minibatch(self, minibatch):

        states = []
        target_values = []

        # iterate over memories and build target list
        for memory in minibatch:

            old_state_m, action_m, reward_m, new_state_m = memory

            # predict Q value for old state
            old_qval = self.model.predict(old_state_m, batch_size=1)

            # predict Q value for new state
            new_qval = self.model.predict(new_state_m, batch_size=1)

            # choose max Q value predicted for the new state
            max_new_qval = np.max(new_qval)

            y = np.zeros((1, 3))
            y[:] = old_qval[:]

            if reward_m != -4000:
                update = (reward_m + (self.gamma * max_new_qval))
            else:
                update = reward_m

            # update Q value with the target value
            y[0][action_m] = update

            states.append(old_state_m.reshape(self.input_dim, ))
            target_values.append(y.reshape(3, ))

        # transform list to array to be passed to the network
        states = np.array(states)
        target_values = np.array(target_values)

        return states, target_values
