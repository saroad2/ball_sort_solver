from collections import deque
import numpy as np


class BallSortGame:

    def __init__(
        self,
        balls_colors_number: int,
        stack_capacity: int,
        extra_stacks: int,
        score_base: float,
        won_reward: float,
        score_gain_reward: float,
        score_loss_penalty: float,
        score_change_rate: float,
        illegal_move_loss: float,
        move_loss: float,
    ):
        self.balls_colors_number = balls_colors_number
        self.stack_capacity = stack_capacity
        self.extra_stacks = extra_stacks
        self.score_base = score_base
        self.won_reward = won_reward
        self.score_gain_reward = score_gain_reward
        self.score_loss_penalty = score_loss_penalty
        self.score_change_rate = score_change_rate
        self.illegal_move_loss = illegal_move_loss
        self.move_loss = move_loss

        self.stacks = [deque() for _ in range(balls_colors_number + extra_stacks)]
        self.duration = 0
        self.fill_stacks()

    @property
    def stacks_number(self):
        return len(self.stacks)

    @property
    def score(self):
        scores_sum = 0
        for stack in self.stacks:
            if len(stack) == 0:
                continue
            i = 1
            while i < len(stack) and stack[i] == stack[0]:
                i += 1
            scores_sum += np.power(self.score_base, i)
        return scores_sum

    @property
    def won(self):
        completed_stacks = [
            i for i in range(self.stacks_number)
            if self.stack_completed(i)
        ]
        return len(completed_stacks) == self.balls_colors_number

    def stack_size(self, stack_index):
        return len(self.stacks[stack_index])

    def stack_remaining_space(self, stack_index):
        return self.stack_capacity - self.stack_size(stack_index)

    def stack_completed(self, stack_index):
        if self.stack_remaining_space(stack_index) != 0:
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
        prev_score = self.score
        moved_ball = self.top_ball(from_index)
        if moved_ball is None:
            return -self.illegal_move_loss, True
        if self.stack_size(to_index) == self.stack_capacity:
            return -self.illegal_move_loss, True
        to_top_ball = self.top_ball(to_index)
        if to_top_ball is not None and to_top_ball != moved_ball:
            return -self.illegal_move_loss, True
        self.duration += 1
        self.stacks[to_index].append(self.stacks[from_index].pop())
        if self.won:
            return self.won_reward, True
        reward = self.move_reward(
            current_score=self.score, prev_score=prev_score
        )
        return reward, False

    def move_reward(self, current_score, prev_score):
        if current_score > prev_score:
            return self.score_gain_reward * np.exp(
                self.score_change_rate * (current_score - prev_score)
            )
        if current_score < prev_score:
            return -self.score_loss_penalty * np.exp(
                self.score_change_rate * (prev_score - current_score)
            )
        return -self.move_loss

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
        self.duration = 0
        for stack in self.stacks:
            stack.clear()
        self.fill_stacks()

    def __repr__(self):
        return "\n".join(f"\t{i}) {list(stack)}" for i, stack in enumerate(self.stacks))
