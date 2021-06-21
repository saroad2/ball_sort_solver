import numpy as np


class ReplayBuffer:
    def __init__(self, memory_size, batch_size, state_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.memory_size, state_size))
        self.new_state_memory = np.zeros((self.memory_size, state_size))
        self.action_memory = np.zeros(self.memory_size, dtype=np.int)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def __len__(self):
        return min(self.mem_cntr, self.memory_size)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.memory_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones