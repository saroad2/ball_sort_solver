import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout


class CriticNetwork(keras.Model):

    def __init__(
        self, inner_layers_neurons, dropout_rate, name='critic'
    ):
        super(CriticNetwork, self).__init__()

        self.inner_layers_neurons = inner_layers_neurons
        self.dropout_rate = dropout_rate
        self.model_name = name

        self.fc1 = Dense(inner_layers_neurons, activation='relu')
        self.do1 = Dropout(dropout_rate)
        self.fc2 = Dense(inner_layers_neurons, activation='relu')
        self.do2 = Dropout(dropout_rate)
        self.q = Dense(1, activation=None)

    def call(self, state_action):
        action_value = self.fc1(state_action)
        action_value = self.do1(action_value)
        action_value = self.fc2(action_value)
        action_value = self.do2(action_value)

        q = self.q(action_value)

        return q

    def get_config(self):
        return {
            "inner_layers_neurons": self.inner_layers_neurons,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ActorNetwork(keras.Model):
    def __init__(
        self,
        inner_layers_neurons,
        dropout_rate,
        action_size,
        name='actor',
    ):
        super(ActorNetwork, self).__init__()

        self.inner_layers_neurons = inner_layers_neurons
        self.dropout_rate = dropout_rate
        self.action_size = action_size
        self.model_name = name

        self.fc1 = Dense(inner_layers_neurons, activation='relu')
        self.do1 = Dropout(dropout_rate)
        self.fc2 = Dense(inner_layers_neurons, activation='relu')
        self.do2 = Dropout(dropout_rate)
        self.mu = Dense(action_size, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.do1(prob)
        prob = self.fc2(prob)
        prob = self.do2(prob)

        mu = self.mu(prob)

        return mu

    def get_config(self):
        return {
            "inner_layers_neurons": self.inner_layers_neurons,
            "dropout_rate": self.dropout_rate,
            "action_size": self.action_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
