from collections import deque
import numpy as np

from ball_sort_solver.exceptions import IllegalMove


class BallSortGame:

    def __init__(
        self,
        balls_colors_number,
        stack_capacity,
        extra_stacks
    ):
        self.balls_colors_number = balls_colors_number
        self.stack_capacity = stack_capacity
        self.extra_stacks = extra_stacks
        self.stacks = [deque() for _ in range(balls_colors_number + extra_stacks)]
        self.fill_stacks()

    @property
    def stacks_number(self):
        return len(self.stacks)

    @property
    def score(self):
        return len(
            [
                i for i in range(self.stacks_number)
                if self.stack_completed(i)
             ]
        )

    @property
    def won(self):
        return self.score == self.balls_colors_number

    def stack_size(self, stack_index):
        return len(self.stacks[stack_index])

    def stack_completed(self, stack_index):
        if self.stack_size(stack_index) != self.stack_capacity:
            return False
        return all(
            self.stacks[stack_index][i] == self.stacks[stack_index][0]
            for i in range(self.stack_capacity)
        )

    def top_ball(self, stack_index):
        if self.stack_size(stack_index) == 0:
            return None
        return self.stacks[stack_index][-1]

    def move(self, from_index, to_index):
        moved_ball = self.top_ball(from_index)
        if moved_ball is None:
            raise IllegalMove("Can't move ball from an empty stack")
        if self.stack_size(to_index) == self.stack_capacity:
            raise IllegalMove("Can't move ball to full stack")
        to_top_ball = self.top_ball(to_index)
        if to_top_ball is not None and to_top_ball != moved_ball:
            raise IllegalMove(
                "Can't move ball to stack with top ball with different color"
            )
        self.stacks[to_index].append(self.stacks[from_index].pop())

    def fill_stacks(self):
        remaining_balls = {
            ball_color: self.stack_capacity
            for ball_color in range(1, self.balls_colors_number + 1)
        }
        for i in range(self.balls_colors_number):
            for _ in range(self.stack_capacity):
                ball_color = np.random.choice(list(remaining_balls.keys()))
                self.stacks[i].append(ball_color)
                remaining_balls[ball_color] -= 1
                if remaining_balls[ball_color] == 0:
                    del remaining_balls[ball_color]

    def reset(self):
        for stack in self.stacks:
            stack.clear()
        self.fill_stacks()

    def __repr__(self):
        repr_string = f"Score: {self.score}\n"
        repr_string += "\n".join(f"\t{list(stack)}" for stack in self.stacks)
        return repr_string
