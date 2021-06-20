from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_mutator import BallSortMutator
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.buffer import ReplayBuffer
from ball_sort_solver.networks import build_network
from ball_sort_solver.noise import Noise

EPSILON = 1e-5


class BallSortLearner:

    def __init__(
        self,
        game: BallSortGame,
        state_getter: BallSortStateGetter,
        gamma: float,
        tau: float,
        tau_decay: float,
        start_noise: float,
        noise_decay: float,
        min_noise: float,
        max_duration: int,
        buffer_capacity: int,
        batch_size: int,
        actor_learning_rate: int,
        actor_inner_layers_neurons: int,
        actor_inner_layers: int,
        actor_dropout_rate: float,
        critic_learning_rate: int,
        critic_inner_layers_neurons: int,
        critic_inner_layers: int,
        critic_dropout_rate: float,
        scalars_window: int,
        logs_dir: Optional[Path] = None,
        checkpoints_dir: Optional[Path] = None,
    ):
        self.game = game
        self.state_getter = state_getter

        self.gamma = gamma
        self.tau = tau
        self.tau_decay = tau_decay
        self.max_duration = max_duration
        self.scalars_window = scalars_window

        self.noise = Noise(
            mean=np.zeros(self.actions_size),
            std_deviation=start_noise,
            noise_decay=noise_decay,
            min_noise=min_noise,
        )
        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir
        if checkpoints_dir is not None:
            self.actor_checkpoint = self.checkpoints_dir / "actor_ddpg.h5"
            self.critic_checkpoint = self.checkpoints_dir / "critic_ddpg.h5"
            self.target_actor_checkpoint = self.checkpoints_dir / "target_actor_ddpg.h5"
            self.target_critic_checkpoint = self.checkpoints_dir / "target_critic_ddpg.h5"
        self.writer = (
            tf.summary.create_file_writer(str(self.logs_dir))
            if self.logs_dir is not None
            else None
        )

        self.actor = self.build_actor_model(
            dropout_rate=actor_dropout_rate,
            inner_layers_neurons=actor_inner_layers_neurons,
            inner_layers=actor_inner_layers,
            name="actor"
        )
        self.critic = self.build_critic_model(
            dropout_rate=critic_dropout_rate,
            inner_layers_neurons=critic_inner_layers_neurons,
            inner_layers=critic_inner_layers,
            name="critic",
        )

        self.target_actor = self.build_actor_model(
            dropout_rate=actor_dropout_rate,
            inner_layers_neurons=actor_inner_layers_neurons,
            inner_layers=actor_inner_layers,
            name="target_actor"
        )
        self.target_critic = self.build_critic_model(
            dropout_rate=critic_dropout_rate,
            inner_layers_neurons=critic_inner_layers_neurons,
            inner_layers=critic_inner_layers,
            name="critic",
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

        self.model_age = 0
        self.train_history = []

    def build_actor_model(
        self,
        dropout_rate,
        inner_layers_neurons,
        inner_layers,
        name,
    ):
        return build_network(
            input_size=self.state_size,
            inner_layers_neurons=inner_layers_neurons,
            dropout_rate=dropout_rate,
            inner_layers=inner_layers,
            inner_activation="relu",
            output_size=self.actions_size,
            output_activation="tanh",
            name=name,
        )

    def build_critic_model(
        self,
        dropout_rate,
        inner_layers_neurons,
        inner_layers,
        name,
    ):
        return build_network(
            input_size=self.state_size + self.actions_size,
            inner_layers_neurons=inner_layers_neurons,
            dropout_rate=dropout_rate,
            inner_layers=inner_layers,
            inner_activation="relu",
            output_size=1,
            output_activation=None,
            name=name,
        )

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

    def recent_score_span_mean(self, window):
        return self.recent_field_mean(field="score_span", window=window)

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

    def run_episode(self):
        self.game.reset()
        self.noise.update()
        episodic_reward = 0
        initial_score = max_score = self.game.score
        actor_losses = []
        critic_losses = []
        taus = []

        while self.game.duration < self.max_duration:
            prev_state, action, reward, current_state, done = self.make_move()

            self.buffer.store_transition(
                prev_state, action, reward, current_state, done
            )
            episodic_reward += reward

            actor_loss, critic_loss = self.learn()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            tau = self.tau * np.exp(-critic_loss * self.tau_decay)
            self.update_models(tau)
            taus.append(tau)
            max_score = max(max_score, self.game.score)

            # End this episode when `done` is True
            if done:
                break

        index = len(self.train_history)
        self.save_history(
            index=index,
            reward=episodic_reward,
            duration=self.game.duration,
            final_score=self.game.score,
            score_difference=self.game.score - initial_score,
            score_span=max_score - initial_score,
            actor_loss=np.mean(actor_losses),
            critic_loss=np.mean(critic_losses),
            model_age=self.model_age,
            tau=np.mean(taus),
            noise=self.noise.std_dev,
            save_scalars=True,
        )
        self.save_scalars(
            index=index,
            recent_score_span_mean=self.recent_score_span_mean(self.scalars_window),
            recent_score_difference_mean=self.recent_score_difference_mean(
                self.scalars_window
            ),
            recent_final_score_mean=self.recent_final_score_mean(self.scalars_window),
            recent_duration_mean=self.recent_duration_mean(self.scalars_window),
            recent_rewards_mean=self.recent_reward_mean(self.scalars_window),
            recent_actor_loss_mean=self.recent_actor_loss_mean(self.scalars_window),
            recent_critic_loss_mean=self.recent_critic_loss_mean(self.scalars_window),
        )

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state.reshape(-1, *state.shape)))
        action = sampled_actions.numpy() + self.noise()
        action = tf.clip_by_value(action, -1, 1)
        return action

    def action_to_move(self, action):
        from_indices = np.where(action < 0, -action, 0)
        to_indices = np.where(action > 0, action, 0)
        return (
            np.argmax(from_indices),
            np.argmax(to_indices)
        )

    def make_move(self):
        prev_state = self.state_getter.get_state(self.game)
        action = self.policy(prev_state)

        from_index, to_index = self.action_to_move(action)
        reward, done = self.game.move(from_index=from_index, to_index=to_index)
        current_state = self.state_getter.get_state(self.game)
        return prev_state, action, reward, current_state, done

    def update_models(self, tau):
        if tau < EPSILON:
            return
        self.update_model(model=self.actor, target_model=self.target_actor, tau=tau)
        self.update_model(model=self.critic, target_model=self.target_critic, tau=tau)
        self.model_age += 1

    def update_model(self, model, target_model, tau=None):
        if tau is None:
            tau = self.tau
        model_weights, target_weights = model.get_weights(), target_model.get_weights()
        new_weights = []
        for (a, b) in zip(model_weights, target_weights):
            new_weights.append(a * tau + b * (1 - tau))
        target_model.set_weights(new_weights)

    def learn(self):
        if self.buffer.mem_cntr < self.buffer.batch_size:
            return 0, 0

        state, action, reward, new_state, done = self.buffer.sample_buffer()

        mutator = BallSortMutator(
            stacks_number=self.game.stacks_number,
            balls_colors_number=self.game.balls_colors_number
        )

        states = mutator.mutate_state(state)
        states_ = mutator.mutate_state(new_state)
        actions = mutator.mutate_action(action)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(
                self.target_critic(tf.concat([states_, target_actions], axis=1)),
                1
            )
            critic_value = tf.squeeze(
                self.critic(tf.concat([states, actions], axis=1)),
                1
            )
            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(
                tf.concat([states, new_policy_actions], axis=1)
            )
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        return actor_loss, critic_loss

    def save_history(self, index, save_scalars=True, **kwargs):
        self.train_history.append(kwargs)
        if save_scalars:
            self.save_scalars(index=index, **kwargs)

    def save_scalars(self, index, **kwargs):
        if self.writer is None:
            return
        with self.writer.as_default():
            for key, value in kwargs.items():
                tf.summary.scalar(name=key, data=value, step=index)

    def save_models(self):
        self.actor.save_weights(str(self.actor_checkpoint))
        self.target_actor.save_weights(str(self.target_actor_checkpoint))

        self.critic.save_weights(str(self.critic_checkpoint))
        self.target_critic.save_weights(str(self.target_critic_checkpoint))

    def load_models(self):
        self.actor.load_weights(str(self.actor_checkpoint))
        self.target_actor.load_weights(str(self.target_actor_checkpoint))

        self.critic.load_weights(str(self.critic_checkpoint))
        self.target_critic.load_weights(str(self.target_critic_checkpoint))
