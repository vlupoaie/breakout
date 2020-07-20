from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import pickle
import numpy as np
import random
import csv


class DQNAlgorithm:
    def __init__(self, input_dimension, epsilon, gamma, batch_size, observe, replay_memory_dimension,
                 first_layer_dimension, second_layer_dimension, weights_file, counter_observer=0,
                 replay_memory=None, test_flag=False):
        self.input_dimension = input_dimension
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.observe = observe
        self.replay_memory_dimension = replay_memory_dimension
        self.layers_dimensions = [first_layer_dimension, second_layer_dimension]
        self.weights_file = weights_file
        self.counter_observer = counter_observer
        self.counter_replay = 0
        self.counter_over = 0
        self.loss_log = []
        self.store = []
        self.old_state = None
        self.new_state = None
        self.action = None
        self.reward = None
        self.score = 0
        self.test_flag = test_flag
        self.upper_limit = 70

        # restore replay memory or make it a new empty list
        if replay_memory is None:
            replay_memory = []
        else:
            self.replay_memory = replay_memory

        # initialise new layer or load existing one
        self.model = self.new_neural_network_model(self.input_dimension, self.layers_dimensions, self.weights_file)

        # if we're testing the algorithm
        if self.test_flag:
            self.observe = 10 ** 8
            self.epsilon = 0

    @staticmethod
    def new_neural_network_model(input_dimension, layers_size, load_file=None):
        model = Sequential()

        # first layer
        model.add(Dense(layers_size[0], init='lecun_uniform', input_shape=(input_dimension,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # hidden second layer
        model.add(Dense(layers_size[1], init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(3, init='lecun_uniform'))
        model.add(Activation('linear'))

        # add RMS optimizer
        rms_prop = RMSprop()
        model.compile(loss='mse', optimizer=rms_prop)

        # load weights
        if load_file:
            model.load_weights(load_file + ".h5")
        return model

    @classmethod
    def load_from_file(cls, weights_file, parameters_file):
        with open(parameters_file, "rb") as handle:
            parameters = pickle.loads(handle.read())

        input_dimension = parameters['input_dimension']
        epsilon = parameters['epsilon']
        gamma = parameters['gamma']
        batch_size = parameters['batch_size']
        observe = parameters['observe']
        counter_observer = parameters['counter_observer']
        replay_memory_dimension = parameters['replay_memory_dimension']
        layers_dimensions = parameters['layers_dimensions']
        replay_memory = parameters['replay_memory']

        return cls(input_dimension, epsilon, gamma, batch_size, observe, replay_memory_dimension,
                   layers_dimensions[0], layers_dimensions[1], weights_file, counter_observer, replay_memory)


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
    @classmethod
    def load_from_save(cls, filename_to_store_weights, file_to_load):

        filename_to_store_weights = filename_to_store_weights
        f = open(file_to_load, "rb")
        store = pickle.loads(f.read())
        f.close()

        input_dim = 5
        epsilon = store[1]
        gamma = store[7]
        batch_size = store[8]
        observe = store[9]
        replay_memory_dim = 30000
        list_params_layers = [100, 100]
        filename_to_store_weights = filename_to_store_weights

        replay_memory = store[6]
        model = init_neural_net(input_dim, list_params_layers, filename_to_store_weights)
        counter_observer = store[0]

        return cls(input_dim, epsilon, gamma, batch_size, observe, replay_memory_dim, 100, 100,
                   filename_to_store_weights, counter_observer, model, replay_memory)

    def __init__(self, input_dim, epsilon, gamma, batch_size, observe, replay_memory_dim, first_layer_dimension,
                 second_layer_dimension, filename_to_store_weights, counter_observer=0, model=None,
                 replay_memory=list(), test_flag=0):

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.observe = observe
        self.replay_memory_dim = replay_memory_dim
        self.list_params_layers = [first_layer_dimension, second_layer_dimension]
        self.filename_to_store_weights = filename_to_store_weights

        self.replay_memory = replay_memory
        if model == None:
            self.model = init_neural_net(self.input_dim, self.list_params_layers)
        else:
            self.model = model
        self.counter_observer = counter_observer
        self.counter_replay = 0
        self.counter_over = 0
        self.loss_log = []
        self.store = []

        self.old_state = None
        self.new_state = None
        self.action = None
        self.reward = None

        self.score = 0

        self.test_flag = test_flag

        self.upper_limit = 70

        if self.test_flag:
            self.observe = 99999999
            self.epsilon = 0
            self.model = init_neural_net(self.input_dim, self.list_params_layers, self.filename_to_store_weights)

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

    # callback function - am pus training_frames ca parametru ca nr-frame-ului ar trebui sa vina de la tine, am nevoie de el ca sa scad epsilonul sau pot sa folosesc alt numar mare.
    def game_learner(self, state, reward, training_frames=30000):

        state = np.array([state])
        print("==================STATE=============")
        print(str(state))
        print("==================REWARD============")
        print(str(reward))
        self.new_state = state

        self.reward = reward

        self.counter_observer += 1

        # if not initial state
        if self.old_state != None:
            # add to replay memory tuple (s, a, r, s')

            self.replay_memory.append((self.old_state, self.action, self.reward, self.new_state))

            print(
                "Frame: " + str(self.counter_observer) + "------ Reward: " + str(self.reward) + "--------Score: " + str(
                    self.score) + "------ Epsilon: " + str(self.epsilon))

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
                self.model.fit(states, target_values, batch_size=self.batch_size, nb_epoch=1, verbose=0,
                               callbacks=[history])
                self.loss_log.append(history.losses)
                print("---------------Losses: " + str(history.losses))

                # prepare transition to next state
        self.old_state = self.new_state

        # decrement epsilon if it didn't reach 0 yet and if we finished the observation
        if self.epsilon > 0 and self.counter_observer > self.observe:
            self.epsilon -= (1 / training_frames)

        # save losses log if training is over
        if self.counter_observer == training_frames:
            log_results(self.filename_to_store_weights, self.loss_log)

        # if game over, update variables, and return -1 (inexistent action)
        if self.reward == -1 and self.test_flag == 0:

            self.counter_over += 1

            self.reset_variables_if_game_over()

            ok = 0

            if self.counter_observer < self.observe:
                if self.counter_over % 10 == 0:
                    ok = 1
            else:
                if self.counter_over % 10 == 0:
                    ok = 1

            if ok == 1:
                f = open("backup-data", "wb")

                self.store = []
                self.store.append(self.counter_observer)
                self.store.append(self.epsilon)
                self.store.append(self.new_state)
                self.store.append(self.old_state)
                self.store.append(self.action)
                self.store.append(self.reward)
                self.store.append(self.replay_memory)
                self.store.append(self.gamma)
                self.store.append(self.batch_size)
                self.store.append(self.observe)
                self.store.append(self.replay_memory_dim)

                f.write(pickle.dumps(self.store))

                print("Saving data...")

        self.choose_action()

        # save trained network every 1 000 frames
        if self.counter_observer % 1000 == 0 and self.test_flag == 0:
            self.model.save_weights(self.filename_to_store_weights + ".h5", overwrite=True)
            print("Saving model %s - %d" % (self.filename_to_store_weights, self.counter_observer))

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

            if reward_m != -1:
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





