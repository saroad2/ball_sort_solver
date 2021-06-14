from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.buffer import ReplayBuffer
from ball_sort_solver.networks import ActorNetwork, CriticNetwork


class BallSortLearner:

    def __init__(
        self,
        game: BallSortGame,
        state_getter: BallSortStateGetter,
        gamma: float,
        tau: float,
        start_noise: float,
        noise_decay: float,
        min_noise: float,
        max_duration: int,
        buffer_capacity: int,
        batch_size: int,
        actor_learning_rate: int,
        actor_inner_layer_neurons: int,
        actor_dropout_rate: float,
        critic_learning_rate: int,
        critic_inner_layer_neurons: int,
        critic_dropout_rate: float,
        output_dir: Optional[Path]
    ):
        self.game = game
        self.state_getter = state_getter

        self.gamma = gamma
        self.tau = tau
        self.noise = start_noise
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.max_duration = max_duration

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.logs_dir = output_dir / "logs"
            self.checkpoints_dir = output_dir / "checkpoints"
            self.writer = tf.summary.create_file_writer(str(self.logs_dir))
        else:
            self.logs_dir = None
            self.checkpoints_dir = None

        self.actor = ActorNetwork(
            inner_layers_neurons=actor_inner_layer_neurons,
            dropout_rate=actor_dropout_rate,
            action_size=self.actions_size,
            chkpt_dir=self.checkpoints_dir
        )
        self.critic = CriticNetwork(
            inner_layers_neurons=critic_inner_layer_neurons,
            dropout_rate=critic_dropout_rate,
            chkpt_dir=self.checkpoints_dir,
        )

        self.target_actor = ActorNetwork(
            inner_layers_neurons=actor_inner_layer_neurons,
            dropout_rate=actor_dropout_rate,
            action_size=self.actions_size,
            chkpt_dir=self.checkpoints_dir,
        )
        self.target_critic = CriticNetwork(
            inner_layers_neurons=critic_inner_layer_neurons,
            dropout_rate=critic_dropout_rate,
            chkpt_dir=self.checkpoints_dir,
        )

        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        )
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        )
        self.target_actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        )
        self.target_critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        )

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.buffer = ReplayBuffer(
            max_size=buffer_capacity,
            batch_size=batch_size,
            state_size=self.state_size,
            action_size=self.actions_size,
        )

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

    def recent_final_score_mean(self, window):
        return self.recent_field_mean(field="final_score", window=window)

    def recent_score_difference_mean(self, window):
        return self.recent_field_mean(field="score_difference", window=window)

    def recent_actor_loss_mean(self, window):
        return self.recent_field_mean(field="actor_loss", window=window)

    def recent_critic_loss_mean(self, window):
        return self.recent_field_mean(field="critic_loss", window=window)

    def recent_field_mean(self, field, window):
        field_values = [
            history_point[field] for history_point in self.train_history
        ]
        if len(field_values) > window:
            field_values = field_values[-window:]
        return np.mean(field_values)

    def update_noise(self):
        if self.noise > self.min_noise:
            self.noise *= np.exp(-self.noise_decay)

    def run_episode(self):
        self.game.reset()
        self.update_noise()
        episodic_reward = 0
        initial_score = self.game.score
        actor_losses = []
        critic_losses = []

        while self.game.duration < self.max_duration:
            prev_state, action, reward, current_state, done = self.make_move()

            self.buffer.store_transition(
                prev_state, action, reward, current_state, done
            )
            episodic_reward += reward

            actor_loss, critic_loss = self.learn()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # End this episode when `done` is True
            if done:
                break

        self.update_models()
        self.save_history(
            reward=episodic_reward,
            duration=self.game.duration,
            final_score=self.game.score,
            score_difference=self.game.score - initial_score,
            actor_loss=np.mean(actor_losses),
            critic_loss=np.mean(critic_losses)
        )

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state.reshape(-1, *state.shape)))
        noise = np.random.normal(scale=self.noise, size=(self.actions_size,))

        action = sampled_actions.numpy() + noise
        action = tf.clip_by_value(action, -1, 1)
        return action

    def make_move(self):
        prev_state = self.state_getter.get_state(self.game)
        action = self.policy(prev_state)

        from_index, to_index = np.argmin(action), np.argmax(action)
        reward, done = self.game.move(from_index=from_index, to_index=to_index)
        current_state = self.state_getter.get_state(self.game)
        return prev_state, action, reward, current_state, done

    def update_models(self):
        self.update_model(model=self.actor, target_model=self.target_actor)
        self.update_model(model=self.critic, target_model=self.target_critic)

    def update_model(self, model, target_model):
        model_weights, target_weights = model.get_weights(), target_model.get_weights()
        new_weights = []
        for (a, b) in zip(model_weights, target_weights):
            new_weights.append(a * self.tau + b * (1 - self.tau))
        target_model.set_weights(new_weights)

    def learn(self):
        if self.buffer.mem_cntr < self.buffer.batch_size:
            return 0, 0

        state, action, reward, new_state, done = self.buffer.sample_buffer()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        return actor_loss, critic_loss

    def save_history(self, **kwargs):
        index = len(self.train_history)
        self.train_history.append(kwargs)
        if self.writer is None:
            return
        with self.writer.as_default():
            for key, value in kwargs.items():
                tf.summary.scalar(name=key, data=value, step=index)
