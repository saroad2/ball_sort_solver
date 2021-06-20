import numpy as np
import itertools


class BallSortMutator:

    def __init__(self, stacks_number, balls_colors_number):
        self.stacks_permutation = np.random.permutation(stacks_number)
        self.balls_permutation = np.random.permutation(balls_colors_number)
        self.state_permutation = list(
            itertools.chain.from_iterable(
                [
                    (balls_colors_number * stack_index + self.balls_permutation).tolist()
                    for stack_index in self.stacks_permutation
                ]
            )
        )

    def mutate_state(self, states):
        return states[:, self.state_permutation]

    def mutate_action(self, action):
        return action[:, self.stacks_permutation]
