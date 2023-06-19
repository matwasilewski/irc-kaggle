from timeit import timeit

import click

from src.irc_kaggle.pipelines import (
    tune_hyperparameters_on_greeks,
    tune_hyperparameters_on_no_greeks,
)


@click.command()
@click.option(
    '--artefact_dir_path',
    required=True,
    type=str,
    help='The path of the directory where the output pickle files will be stored.',
)
@click.option(
    '--greeks',
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help='Use Epsilon from Greeks',
)
def run_tune_hyperparameters(artefact_dir_path, greeks):
    """
    Command Line Interface for tuning hyperparameters on the Greeks dataset.
    """
    if greeks:
        timeit(tune_hyperparameters_on_no_greeks(artefact_dir_path))
    else:
        timeit(tune_hyperparameters_on_greeks(artefact_dir_path))


if __name__ == '__main__':
    run_tune_hyperparameters()
