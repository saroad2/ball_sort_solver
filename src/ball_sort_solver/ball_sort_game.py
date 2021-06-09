from collections import deque
import numpy as np


class BallSortGame:

    def __init__(
        self,
        balls_colors_number,
        stack_size,
        extra_stacks
    ):
        self.balls_colors_number = balls_colors_number
        self.stack_size = stack_size
        self.extra_stacks = extra_stacks
        self.stacks = [deque() for _ in range(balls_colors_number + extra_stacks)]
        self.fill_stacks()

    def fill_stacks(self):
        remaining_balls = {
            ball_color: self.stack_size
            for ball_color in range(1, self.balls_colors_number + 1)
        }
        for i in range(self.balls_colors_number):
            for _ in range(self.stack_size):
                ball_color = np.random.choice(list(remaining_balls.keys()))
                self.stacks[i].append(ball_color)
                remaining_balls[ball_color] -= 1
                if remaining_balls[ball_color] == 0:
                    del remaining_balls[ball_color]

    def __repr__(self):
        return "\n".join(str(stack) for stack in self.stacks)
