import click
from src.irc_kaggle.pipelines import tune_hyperparameters_on_greeks


@click.command()
@click.option('--artefact_dir_path',
              prompt='Path to the directory to store the artefacts',
              help='The path of the directory where the output pickle files will be stored.')
def run_tune_hyperparameters_on_greeks(artefact_dir_path):
    """
    Command Line Interface for tuning hyperparameters on the Greeks dataset.
    """
    tune_hyperparameters_on_greeks(artefact_dir_path)


if __name__ == '__main__':
    run_tune_hyperparameters_on_greeks()
