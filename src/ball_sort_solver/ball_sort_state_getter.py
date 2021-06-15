import numpy as np

from ball_sort_solver.ball_sort_game import BallSortGame


class BallSortStateGetter:

    @classmethod
    def size(cls, game: BallSortGame):
        return game.stacks_number * game.balls_colors_number

    @classmethod
    def get_state(cls, game: BallSortGame):
        state = np.zeros(shape=(game.stacks_number, game.balls_colors_number))
        for i, stack in enumerate(game.stacks):
            for j, color in enumerate(stack, start=1):
                state[i, color - 1] = j
        return state.flatten() / game.stack_capacity
