"""
Contains helper function for the Typer CLI application for reading
configuration and writing output files.
"""

from typing import Any
from pathlib import Path
import yaml

def read_config(
    config_filepath: Path, default_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Reads config.yaml to dictionary and sets default parameters where no values
    are specified within the file.

    Parameters
    -----------
    config_filepath
        Filepath to config.yaml file.
    default_params
        Default parameters.

    Returns
    --------
    Dictionary containing configuration variables for the CLI train pipeline.
    """
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    for k, v in default_params.items():
        if k not in config or config[k] is None:
            config[k] = v

    return config


def write_performance_scores(
    performance_scores_filepath: Path,
    timestamp: str,
    classifier_used: str,
    performance_scores: dict[str, float | str],
) -> None:
    """
    Writes performance scores achieved by the classifier to specified file.

    Parameters
    -----------
    performance_scores_filepath
        Filepath to output file.
    timestamp
        Timestamp of file creation.
    classifier_used
        Name of the classifier whose performance scores are described.
    performance_scores
        Performance scores computed for the classifier.
    """
    with open(performance_scores_filepath, "w") as f:
        f.write(f"Created at: {timestamp}\n\n")
        f.write(f"Classifier used: {classifier_used}\n\n")
        f.write("-" * 20 + "\n\n")
        for metric, score in performance_scores.items():
            f.write(f"{metric}{score}\n\n")


def write_predictions(
    predictions_filepath: Path, predictions: list[str], filenames: list[str]
) -> None:
    """
    Writes model class predictions and corresponding filenames to a file in CSV
    format.

    Parameters
    -----------
    predictions_filepath
        Filepath to output file.
    predictions
        Predicted classes for each file containing a spectrum.
    filenames
        Filenames of files containing the spectra whose class was predicted.
    """
    with open(predictions_filepath, "w") as f:
        f.write("prediction, filename\n")
        for pred, fn in zip(predictions, filenames):
            f.write(f"{pred}, {fn}\n")
