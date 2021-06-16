import json
import shutil
from pathlib import Path

import click

from ball_sort_solver.ball_sort_game import BallSortGame
from ball_sort_solver.ball_sort_learner import BallSortLearner
from ball_sort_solver.ball_sort_state_getter import BallSortStateGetter
from ball_sort_solver.plot_util import plot_all_field_plots


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
        logs_dir = output_dir / "logs"
        checkpoints_dir = output_dir / "checkpoints"
        plots_dir = output_dir / "plots"
        checkpoints_dir.mkdir()
        plots_dir.mkdir()
    else:
        logs_dir = None
        checkpoints_dir = None
        plots_dir = None
    with open(configuration, mode="r") as fd:
        config_dict = json.load(fd)
    game = BallSortGame(**config_dict["game"])
    state_getter = BallSortStateGetter()
    learner = BallSortLearner(
        game=game,
        state_getter=state_getter,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        **config_dict["learner"]
    )
    train_config = config_dict["train"]
    episodes = train_config["episodes"]
    plot_window = train_config["plot_window"]
    with click.progressbar(length=episodes, show_pos=True, show_percent=False) as bar:
        try:
            for _ in bar:
                learner.run_episode()
                bar.label = (
                    f"Rewards: {learner.recent_reward_mean(plot_window):.2f}, "
                    f"Duration: {learner.recent_duration_mean(plot_window):.2f}, "
                    f"Final score: {learner.recent_final_score_mean(plot_window):.2f}, "
                    f"Score diff: {learner.recent_score_difference_mean(plot_window):.2f}, "
                    f"Score span: {learner.recent_score_span_mean(plot_window):.2f}, "
                    f"Actor loss: {learner.recent_actor_loss_mean(plot_window):.2e}, "
                    f"Critic loss: {learner.recent_critic_loss_mean(plot_window):.2e}"
                )
        except KeyboardInterrupt:
            click.echo()
            if not click.confirm(
                "Training was interrupted. Would you like to save results?",
                default=False,
            ):
                click.echo("Training aborted!")
                return

    if plots_dir is None:
        return
    click.echo("Saving Models...")
    learner.save_models()
    click.echo("Done!")
    click.echo("Saving plots...")
    plot_all_field_plots(
        history=learner.train_history,
        field="final_score",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="score_difference",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="score_span",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="duration",
        output_dir=plots_dir,
        plot_window=plot_window,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="reward",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="actor_loss",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    plot_all_field_plots(
        history=learner.train_history,
        field="critic_loss",
        output_dir=plots_dir,
        plot_window=plot_window,
        is_float=True,
    )
    click.echo("Done!")


if __name__ == '__main__':
    ball_sort_solver_cli()
