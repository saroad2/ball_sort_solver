import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, inner_layers_neurons, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(inner_layers_neurons, activation='relu')
        self.fc2 = Dense(inner_layers_neurons, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(
        self, inner_layers_neurons, action_size, name='actor', chkpt_dir='tmp/ddpg'
    ):
        super(ActorNetwork, self).__init__()

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(inner_layers_neurons, activation='relu')
        self.fc2 = Dense(inner_layers_neurons, activation='relu')
        self.mu = Dense(action_size, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu
