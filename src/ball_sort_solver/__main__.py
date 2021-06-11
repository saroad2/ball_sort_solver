import json
import shutil
from pathlib import Path

import click

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_learner import BallSortLearner
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter


@click.group()
def ball_sort_solver_cli():
    """
    Ball Sort game solver
    """


@ball_sort_solver_cli.command("play")
@click.option(
    "-c", "--configuration",
    type=click.Path(dir_okay=False, exists=True),
)
def play_ball_sort(configuration):
    configuration = (
        Path(configuration)
        if configuration is not None
        else Path.cwd() / "configuration.json"
    )
    with open(configuration, mode="r") as fd:
        config_dict = json.load(fd)
    game = BallSortGame(**config_dict["game"])
    done = False
    rewards_sum = 0
    while not done:
        click.echo(f"Score: {game.score}, Rewards: {rewards_sum}")
        click.echo(game)
        from_index = click.prompt("From index", type=int)
        to_index = click.prompt("To index", type=int)
        reward, done = game.move(from_index, to_index)
        rewards_sum += reward
    if game.won:
        click.echo("Game won!")
    else:
        click.echo("Game lost...")
    click.echo(f"Score: {game.score}, Rewards: {rewards_sum}")


@ball_sort_solver_cli.command("train")
@click.option(
    "-c", "--configuration",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False),
)
def train_ball_sort(configuration, output_dir):
    configuration = (
        Path(configuration)
        if configuration is not None
        else Path.cwd() / "configuration.json"
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    with open(configuration, mode="r") as fd:
        config_dict = json.load(fd)
    game = BallSortGame(**config_dict["game"])
    state_getter = BallSortStateGetter()
    learner = BallSortLearner(
        game=game,
        state_getter=state_getter,
        output_dir=output_dir,
        **config_dict["learner"]
    )
    train_config = config_dict["train"]
    episodes = train_config["episodes"]
    plot_window = train_config["plot_window"]
    with click.progressbar(length=episodes, show_pos=True, show_percent=False) as bar:
        for _ in bar:
            learner.run_episode()
            bar.label = (
                f"Rewards: {learner.recent_reward_mean(plot_window):.2f}, "
                f"Duration: {learner.recent_duration_mean(plot_window):.2f}, "
                f"Final score: {learner.recent_final_score_mean(plot_window):.2f}, "
                f"Max score: {learner.recent_max_score_mean(plot_window):.2f}, "
                f"Actor loss: {learner.recent_actor_loss_mean(plot_window):.2f}, "
                f"Critic loss: {learner.recent_critic_loss_mean(plot_window):.2f}, "
                f"Noise: {learner.noise:.2f}"
            )


if __name__ == '__main__':
    ball_sort_solver_cli()
