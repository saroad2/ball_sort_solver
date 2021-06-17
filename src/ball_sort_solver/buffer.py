import numpy as np


class ReplayBuffer:
    def __init__(self, memory_size, batch_size, state_size, action_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.current_states = np.empty((0, state_size))
        self.actions = np.empty((0, action_size))
        self.rewards = np.empty(0)
        self.new_states = np.empty((0, state_size))
        self.done = np.empty(0, dtype=np.bool)

    def __len__(self):
        return self.current_states.shape[0]

    def store_transition(self, current_state, action, reward, new_state, done):
        self.current_states = np.append(
            self.current_states,
            current_state.reshape((-1, *current_state.shape)),
            axis=0
        )
        self.actions = np.append(
            self.actions,
            action.reshape((-1, *action.shape)),
            axis=0)
        self.rewards = np.append(self.rewards, np.array([reward]), axis=0)
        self.new_states = np.append(
            self.new_states,
            new_state.reshape(-1, *new_state.shape),
            axis=0
        )
        self.done = np.append(self.done, np.array([done]), axis=0)

        if self.current_states.shape[0] > self.memory_size:
            self.current_states = self.current_states[-self.memory_size:]
            self.actions = self.actions[-self.memory_size:]
            self.rewards = self.rewards[-self.memory_size:]
            self.new_states = self.new_states[-self.memory_size:]
            self.done = self.done[-self.memory_size:]

    def sample_buffer(self, p=0):
        max_mem = min(len(self), self.memory_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False, p=p)

        states = self.current_states[batch]
        states_ = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.done[batch]

        return states, actions, rewards, states_, dones
