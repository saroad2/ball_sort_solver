import numpy as np
import tensorflow as tf

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.exceptions import IllegalMove


class BallSortLearner:

    def __init__(
        self,
        game: BallSortGame,
        state_getter: BallSortStateGetter,
        gamma: float,
        tau: float,
        noise_std: float,
        won_reward: float,
        score_gain_reward: float,
        score_loss_penalty: float,
        illegal_move_loss: float,
        move_loss: float,
        buffer_capacity: int,
        batch_size: int,
        actor_learning_rate: int,
        critic_learning_rate: int,
        actor_inner_layer_neurons: int,
        critic_inner_layer_neurons: int,
    ):
        self.game = game
        self.state_getter = state_getter

        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

        self.won_reward = won_reward
        self.score_gain_reward = score_gain_reward
        self.score_loss_penalty = score_loss_penalty
        self.illegal_move_loss = illegal_move_loss
        self.move_loss = move_loss

        self.actor = self.create_actor_model(actor_inner_layer_neurons)
        self.critic = self.create_critic_model(critic_inner_layer_neurons)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

        self.target_actor = self.create_actor_model(actor_inner_layer_neurons)
        self.target_critic = self.create_critic_model(critic_inner_layer_neurons)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, self.actions_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))

        self.train_history = []

    @property
    def state_size(self):
        return self.state_getter.size(self.game)

    @property
    def actions_size(self):
        return self.game.stacks_number

    @property
    def last_train_history(self):
        if len(self.train_history) == 0:
            return None
        return self.train_history[-1]

    def recent_reward_mean(self, window):
        return self.recent_field_mean(field="reward", window=window)

    def recent_duration_mean(self, window):
        return self.recent_field_mean(field="duration", window=window)

    def recent_score_mean(self, window):
        return self.recent_field_mean(field="score", window=window)

    def recent_actor_loss_mean(self, window):
        return self.recent_field_mean(field="actor_loss", window=window)

    def recent_critic_loss_mean(self, window):
        return self.recent_field_mean(field="critic_loss", window=window)

    def recent_field_mean(self, field, window):
        field_values = [
            history_point[field] for history_point in self.train_history
        ]
        if len(field_values) > window:
            field_values = field_values[window:]
        return np.mean(field_values)


    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        return actor_loss, critic_loss

    def run_episode(self):
        self.game.reset()
        episodic_reward = 0
        count = 0
        actor_losses = []
        critic_losses = []

        while True:
            count += 1

            prev_state, action, reward, current_state, done = self.make_move()

            self.record(prev_state, action, reward, current_state)
            episodic_reward += reward

            actor_loss, critic_loss = self.learn()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            self.update_model(model=self.actor, target_model=self.target_actor)
            self.update_model(model=self.critic, target_model=self.target_critic)

            # End this episode when `done` is True
            if done:
                break

        self.save_history(
            reward=episodic_reward,
            duration=count,
            score=self.game.score,
            actor_loss=np.mean(actor_losses),
            critic_loss=np.mean(critic_losses)
        )

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state.reshape(-1, *state.shape)))
        noise = np.random.normal(scale=self.noise_std, size=(self.actions_size,))

        return sampled_actions.numpy() + noise

    def make_move(self):
        prev_state = self.state_getter.get_state(self.game)
        action = self.policy(prev_state)
        prev_score = self.game.score

        from_index, to_index = np.argmin(action), np.argmax(action)
        try:
            self.game.move(from_index=from_index, to_index=to_index)
        except IllegalMove:
            reward = -self.illegal_move_loss
            return prev_state, action, reward, prev_state, True
        current_state = self.state_getter.get_state(self.game)
        if self.game.won:
            done = True
            reward = self.won_reward
        else:
            done = False
            reward = self.move_reward(
                current_score=self.game.score, prev_score=prev_score
            )
        return prev_state, action, reward, current_state, done

    def move_reward(self, current_score, prev_score):
        if current_score > prev_score:
            return current_score * self.score_gain_reward
        if current_score < prev_score:
            return -current_score * self.score_loss_penalty
        return -self.move_loss

    def update_model(self, model, target_model):
        model_weights, target_weights = target_model.get_weights(), model.get_weights()
        new_weights = []
        for (a, b) in zip(model_weights, target_weights):
            new_weights.append(a * self.tau + b * (1 - self.tau))
        target_model.set_weights(new_weights)

    def record(self, prev_state, action, reward, current_state):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = prev_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = current_state

        self.buffer_counter += 1

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def save_history(self, **kwargs):
        self.train_history.append(kwargs)

    def create_actor_model(
        self,
        actor_inner_layer_neurons,
    ):
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=self.state_size),
                tf.keras.layers.Dense(
                    actor_inner_layer_neurons,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.he_uniform,
                ),
                tf.keras.layers.Dense(
                    actor_inner_layer_neurons,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.he_uniform
                ),
                tf.keras.layers.Dense(
                    self.actions_size,
                    activation="tanh",
                    kernel_initializer=tf.keras.initializers.he_uniform)
            ]
        )

    def create_critic_model(
        self,
        critic_inner_layer_neurons,
    ):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        state_out = tf.keras.layers.Dense(16, activation="relu")(state_input)
        state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.actions_size,))
        action_out = tf.keras.layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])

        out = tf.keras.layers.Dense(
            critic_inner_layer_neurons,
            activation="relu",
        )(concat)
        out = tf.keras.layers.Dense(
            critic_inner_layer_neurons,
            activation="relu",
        )(out)
        outputs = tf.keras.layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model
