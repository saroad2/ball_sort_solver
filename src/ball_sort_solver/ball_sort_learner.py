import numpy as np
import tensorflow as tf

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.buffer import ReplayBuffer
from ball_sort_solver.stat_util import safe_mean


EPSILON = 1e-5


class BallSortLearner:

    def __init__(
        self,
        game: BallSortGame,
        state_getter: BallSortStateGetter,
        max_duration,
        neurons_per_layer,
        inner_layers,
        learning_rate,
        drop_rate,
        discount,
        epsilon_decrease_rate,
        min_epsilon,
        buffer_capacity,
        batch_size,
        logs_dir=None,
    ):
        self.game = game
        self.state_getter = state_getter
        self.train_history = []
        self.model_age = 0
        self.epsilon = 1

        self.max_duration = max_duration

        self.neurons_per_layer = neurons_per_layer
        self.inner_layers = inner_layers
        self.learning_rate = learning_rate
        self.discount = discount
        self.drop_rate = drop_rate
        self.epsilon_decrease_rate = epsilon_decrease_rate
        self.min_epsilon = min_epsilon

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()

        self.replay_memory = ReplayBuffer(
            memory_size=buffer_capacity,
            batch_size=batch_size,
            state_size=self.state_size,
        )
        self.logs_dir = logs_dir
        self.writer = (
            None if logs_dir is None
            else tf.summary.create_file_writer(str(self.logs_dir))
        )

    @property
    def state_size(self):
        return self.state_getter.size(self.game)

    @property
    def action_size(self):
        return self.game.stacks_number ** 2

    def update_epsilon(self):
        if self.epsilon <= self.min_epsilon:
            return
        self.epsilon *= self.epsilon_decrease_rate

    def run_episode(self):
        self.game.reset()
        accuracies = []
        losses = []
        rewards_sum = 0
        start_score = max_score = self.game.score
        done = False
        while not done and self.game.duration < self.max_duration:
            reward, done = self.make_move()
            rewards_sum += reward
            max_score = max(max_score, self.game.score)
            # Start training only if certain number of samples is already saved
            if len(self.replay_memory) < self.replay_memory.batch_size:
                continue
            history = self.train()
            history = history.history if history is not None else None
            accuracies.append(history["accuracy"][0] if history is not None else 0)
            losses.append(history["loss"][0] if history is not None else 0)
        self.save_history(
            score=self.game.score,
            score_span=max_score - start_score,
            score_difference=self.game.score - start_score,
            duration=self.game.duration,
            reward=rewards_sum,
            accuracy=safe_mean(accuracies),
            loss=safe_mean(losses),
            epsilon=self.epsilon,
            model_age=self.model_age,
        )

    def make_move(self):
        current_state = self.state_getter.get_state(self.game)
        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = self.get_action(current_state)
        else:
            # Get random action
            action = np.random.randint(0, self.action_size)
        from_index, to_index = action // self.game.stacks_number, action % self.game.stacks_number
        reward, done = self.game.move(from_index=from_index, to_index=to_index)
        new_state = self.state_getter.get_state(self.game)
        self.replay_memory.store_transition(
            state=current_state,
            action=action,
            reward=reward,
            state_=new_state,
            done=done,
        )
        return reward, done

    def train(self):

        # Get a minibatch of random samples from memory replay table
        states, actions, rewards, states_, dones = self.replay_memory.sample_buffer()

        # Get current states from minibatch, then query NN model for Q values
        current_qs_list = self.model.predict(states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        future_qs_list = self.target_model.predict(states_)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index in range(self.replay_memory.batch_size):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not dones[index]:
                max_future_q = np.max(future_qs_list[index])
                new_q = rewards[index] + self.discount * max_future_q
            else:
                new_q = rewards[index]

            # Update Q value for given state
            current_qs = current_qs_list[index, :]
            current_qs[actions[index]] = new_q

            # And append to our training data
            X.append(states[index, :])
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        return self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=self.replay_memory.batch_size,
            verbose=0,
            shuffle=False
        )

    def get_q(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def get_action(self, state):
        return np.argmax(self.get_q(state))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.model_age += 1

    def recent_rewards_mean(self, n):
        return self.recent_field_mean("reward", n)

    def recent_score_mean(self, n):
        return self.recent_field_mean("score", n)

    def recent_score_span_mean(self, n):
        return self.recent_field_mean("score_span", n)

    def recent_score_difference_mean(self, n):
        return self.recent_field_mean("score_difference", n)

    def recent_loss_mean(self, n):
        return self.recent_field_mean("loss", n)

    def recent_accuracy_mean(self, n):
        return self.recent_field_mean("accuracy", n)

    def recent_duration_mean(self, n):
        return self.recent_field_mean("duration", n)

    def recent_field_mean(self, field, n):
        field_values = [history_point[field] for history_point in self.train_history]
        if len(field_values) > n:
            field_values = field_values[-n:]
        return np.mean(field_values)

    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.state_size, )))
        for _ in range(self.inner_layers):
            model.add(
                tf.keras.layers.Dense(
                    self.neurons_per_layer,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.he_uniform,
                )
            )
            if self.drop_rate > EPSILON:
                model.add(tf.keras.layers.Dropout(self.drop_rate))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        return model

    def save_model(self, output_path):
        self.target_model.save_weights(output_path)

    def load_model(self, model_path):
        self.model.load_weights(model_path)
        self.target_model.load_weights(model_path)

    def save_history(self, **kwargs):
        index = len(self.train_history)
        self.train_history.append(kwargs)
        if self.writer is None:
            return
        with self.writer.as_default():
            for key, value in kwargs.items():
                tf.summary.scalar(key, value, step=index)
