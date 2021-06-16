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
        self.reset()

    @property
    def stacks_number(self):
        return len(self.stacks)

    @property
    def score(self):
        scores_sum = 0
        for i in range(self.stacks_number):
            scores_sum += self.stack_score(i)
        return scores_sum

    @property
    def won(self):
        completed_stacks = [
            i for i in range(self.stacks_number)
            if self.stack_completed(i)
        ]
        return len(completed_stacks) == self.balls_colors_number

    @property
    def max_stack_score(self):
        return np.power(self.score_base, self.stack_capacity - 1)

    @property
    def max_possible_score(self):
        return self.balls_colors_number * self.max_stack_score

    def stack_size(self, stack_index):
        return len(self.stacks[stack_index])

    def stack_remaining_space(self, stack_index):
        return self.stack_capacity - self.stack_size(stack_index)

    def stack_score(self, stack_index):
        if self.stack_size(stack_index) == 0:
            return 0
        i = 0
        while (
            i + 1 < self.stack_size(stack_index)
            and self.stacks[stack_index][i + 1] == self.bottom_ball(stack_index)
        ):
            i += 1
        if i == 0:
            return 0
        return np.power(self.score_base, i)

    def stack_completed(self, stack_index):
        if self.stack_remaining_space(stack_index) != 0:
            return False
        return all(
            self.stacks[stack_index][i] == self.stacks[stack_index][0]
            for i in range(self.stack_capacity)
        )

    def bottom_ball(self, stack_index):
        if self.stack_size(stack_index) == 0:
            return None
        return self.stacks[stack_index][0]

    def top_ball(self, stack_index):
        if self.stack_size(stack_index) == 0:
            return None
        return self.stacks[stack_index][-1]

    def move(self, from_index, to_index):
        prev_score = self.score
        if not self.is_legal_move(from_index=from_index, to_index=to_index):
            return -self.illegal_move_loss, True
        self.duration += 1
        self.stacks[to_index].append(self.stacks[from_index].pop())
        if self.won:
            return self.won_reward, True
        reward = self.move_reward(
            current_score=self.score, prev_score=prev_score
        )
        return reward, False

    def is_legal_move(self, from_index, to_index):
        if from_index == to_index:
            return False
        moved_ball = self.top_ball(from_index)
        if moved_ball is None:
            return False
        if self.stack_size(to_index) == self.stack_capacity:
            return False
        to_top_ball = self.top_ball(to_index)
        if to_top_ball is not None and to_top_ball != moved_ball:
            return False
        return True

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
