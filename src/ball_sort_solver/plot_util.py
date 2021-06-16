import numpy as np
from matplotlib import pyplot as plt

from ball_sort_solver.stat_util import moving_max, moving_mean


def plot_values_history(values, title, xlabel, ylabel, output_path):
    x = np.arange(np.array(values).shape[0])
    fig, ax = plt.subplots()

    ax.plot(x, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(output_path)

    plt.clf()
    plt.close()


def plot_values_histogram(values, title, xlabel, bins, output_path):
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.savefig(output_path)

    plt.clf()
    plt.close()


def plot_field_history(history, output_dir, field):
    field_title = field.replace("_", " ").title()
    plot_values_history(
        values=[history_point[field] for history_point in history],
        title=f"{field_title} History",
        xlabel="Time",
        ylabel=field_title,
        output_path=output_dir / f"{field}_history.png"
    )


def plot_max_field_history(history, output_dir, field):
    field_title = field.replace("_", " ").title()
    plot_values_history(
        values=moving_max([history_point[field] for history_point in history]),
        title=f"Max {field_title} History",
        xlabel="Time",
        ylabel=field_title,
        output_path=output_dir / f"max_{field}_history.png"
    )


def plot_recent_mean_field_history(history, output_dir, field, n):
    field_title = field.replace("_", " ").title()
    plot_values_history(
        values=moving_mean([history_point[field] for history_point in history], n=n),
        title=f"{field_title} Recent Mean History (Window Size={n})",
        xlabel="Time",
        ylabel=field_title,
        output_path=output_dir / f"{field}_recent_mean_history.png"
    )


def plot_int_field_histogram(history, output_dir, field):
    values = [history_point[field] for history_point in history]
    mean_value, std_value = np.mean(values), np.std(values)
    unique, counts = np.unique(values, return_counts=True)
    fig, ax = plt.subplots()

    ax.bar(unique, counts)
    field_title = field.replace("_", " ").title()
    ax.set_title(
        fr"{field_title} Histogram (Mean=${mean_value:.3f}\pm{std_value:.3f}$)"
    )
    ax.set_xlabel(field_title)
    ax.set_ylabel("Count")
    fig.savefig(output_dir / f"{field}_histogram.png")

    plt.clf()
    plt.close()


def plot_float_field_histogram(history, output_dir, field, bins):
    field_title = field.replace("_", " ").title()
    plot_values_histogram(
        values=[history_point[field] for history_point in history],
        title=f"{field_title} Histogram",
        xlabel=field_title,
        output_path=output_dir / f"{field}_histogram.png",
        bins=bins
    )


def plot_all_field_plots(
    history,
    field,
    output_dir,
    plot_window,
    is_float=False,
    bins=50
):
    plot_recent_mean_field_history(
        history=history,
        output_dir=output_dir,
        field=field,
        n=plot_window,
    )
    plot_max_field_history(
        history=history,
        output_dir=output_dir,
        field=field,
    )
    if is_float:
        plot_float_field_histogram(
            history=history,
            output_dir=output_dir,
            field=field,
            bins=bins
        )
    else:
        plot_int_field_histogram(
            history=history,
            output_dir=output_dir,
            field=field,
        )
