import json
from pathlib import Path

import click

from ball_sort_solver.ball_sort_game import BallSortGame
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
    while True:
        click.echo(game)
        from_index = click.prompt("From index", type=int)
        to_index = click.prompt("To index", type=int)
        try:
            game.move(from_index, to_index)
        except IllegalMove as e:
            click.echo(f"You tried to make an illegal move: {e}")


if __name__ == '__main__':
    ball_sort_solver_cli()
