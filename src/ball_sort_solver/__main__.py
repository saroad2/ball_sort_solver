import json
from pathlib import Path

import click

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_learner import BallSortLearner
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.exceptions import IllegalMove


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
    game = BallSortGame(**config_dict)
    while not game.won:
        click.echo(game)
        from_index = click.prompt("From index", type=int)
        to_index = click.prompt("To index", type=int)
        try:
            game.move(from_index, to_index)
        except IllegalMove as e:
            click.echo(f"You tried to make an illegal move: {e}")
    click.echo("Game won!")


@ball_sort_solver_cli.command("train")
@click.option(
    "-c", "--configuration",
    type=click.Path(dir_okay=False, exists=True),
)
def train_ball_sort(configuration):
    configuration = (
        Path(configuration)
        if configuration is not None
        else Path.cwd() / "configuration.json"
    )
    with open(configuration, mode="r") as fd:
        config_dict = json.load(fd)
    game = BallSortGame(**config_dict["game"])
    state_getter = BallSortStateGetter()
    learner = BallSortLearner(
        game=game,
        state_getter=state_getter,
        **config_dict["learner"]
    )
    train_config = config_dict["train"]
    episodes = train_config["episodes"]
    with click.progressbar(length=episodes, show_pos=True, show_percent=False) as bar:
        for _ in bar:
            learner.run_episode()
            history = learner.last_train_history
            bar.label = (
                f"Rewards: {history['reward']:.2f}, "
                f"Duration: {history['duration']}, "
                f"Score: {history['score']}"
            )


if __name__ == '__main__':
    ball_sort_solver_cli()
