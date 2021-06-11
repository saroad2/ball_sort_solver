import numpy as np

from ball_sort_solver.ball_sort_game import BallSortGame


class BallSortStateGetter:

    @classmethod
    def size(cls, game: BallSortGame):
        return game.stacks_number * game.stack_capacity

    @classmethod
    def get_state(cls, game: BallSortGame):
        state = []
        for i in range(game.stacks_number):
            state.extend(game.stacks[i])
            state.extend([0 for _ in range(game.stack_remaining_space(i))])
        state = np.array(state)
        return state / game.balls_colors_number
