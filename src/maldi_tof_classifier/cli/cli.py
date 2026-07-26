"""
A Typer CLI application for training a classifier model on spectral data and
computing predictions for new spectra based on the classifier.

Two commands are available:
- mtc train: trains the classifier based on the data in 'data_train' and
  the parameters specified in 'cli_files/config.yaml' or default parameters.
- mtc predict: Computes class predictions for the files inside
  'data_predict' and writes the labels and corresponding filenames to
  'cli_files/predictions.csv'.
"""

import typer
from pydantic import ValidationError
from pathlib import Path

import numpy as np

from maldi_tof_classifier.extractors import PeakExtractor
from maldi_tof_classifier.extractors import FullSpectraExtractor

from maldi_tof_classifier.pipelines import generate_pipeline

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder
from pyopls import OPLS

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA, TruncatedSVD

# For OPLS-DA
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.multiclass import OneVsRestClassifier

import joblib

from datetime import datetime

from maldi_tof_classifier.cli.cli_helper import (
    read_config,
    write_performance_scores,
    write_predictions,
)

# Initialize the Typer app.
app = typer.Typer()

# Default paths.
TRAIN_DIR = Path(".").parent / "data_train"
PREDICT_DIR = Path(".").parent / "data_predict"

CLI_FILES_DIR = Path(".").parent / "cli_files"
CONFIG_FILENAME = CLI_FILES_DIR / "config.yaml"
TRAINING_RESULTS_FILENAME = CLI_FILES_DIR / "training_performance.txt"
MODEL_SAVEFILE = CLI_FILES_DIR / "pipeline.joblib"
PREDICTIONS_FILENAME = CLI_FILES_DIR / "predictions.csv"

# Default parameters.
train_default_params = {
    "extractor_cls": "PeakExtractor",
    "classifier_cls": "RandomForestClassifier",
    "dim_reducer_cls": None,
    "n_components": 20,
    "test_size": 0.25,
    "oversampling": True,
}

@app.command()
def train():
    """
    Constructs and fits a pipeline object with the parameters specified in
    'cli_files/config.yaml' or default parameters. The pipeline is fitted on
    the measurement files in 'data_train'. The extractor and pipeline model, as
    well as a mapping of the training data's class labels are written to
    'cli_files/pipeline.joblib' and retrievable via the dictionaries keys
    'pipeline' and 'le' respectively. In case OPLS-DA is used for
    classification, additionally, an OPLS transformation object is added to
    'cli_files/pipeline.joblib', which can be retrieved under the key 'opls'.
    The performance scores achieved by the classifier on the test set is
    written to 'cli_files/training_performance.txt'.
    """
    # Check if cli_files directory exists.
    if not CLI_FILES_DIR.is_dir():
        typer.echo(
            f"""Could not find {CLI_FILES_DIR}!\n
                    Please, create the directory with the CLI files!"""
        )
        raise typer.Exit(1)

    # Check if config file exists.
    if not CONFIG_FILENAME.exists():
        typer.echo(
            f"""Could not find {CONFIG_FILENAME}!\n
                    Please, create the file cli_files/config.yaml"""
        )
        raise typer.Exit(1)

    # Check if data_predict directory exists.
    if not TRAIN_DIR.is_dir():
        typer.echo(
            f"""Could not find {TRAIN_DIR}!\n
                    Please, create the directory with the training files!"""
        )
        raise typer.Exit(1)

    # Read config.yaml.
    try:
        config = read_config(CONFIG_FILENAME, train_default_params)
    except FileNotFoundError:
        typer.echo(f"File {CONFIG_FILENAME} not found!")
        raise typer.Exit(1)
    except ValidationError:
        typer.echo(f"Could not parse {CONFIG_FILENAME} correctly!")
        raise typer.Exit(1)


    # Extractor dependent processing.
    if config["extractor_cls"]=="PeakExtractor":
        extractor = PeakExtractor(**config["extractor_params"] or {})

        peaks_dfs, class_labels = extractor.extract_train_data(TRAIN_DIR)
        X_train, X_test, y_train, y_test = train_test_split(
            peaks_dfs,
            class_labels,
            test_size=config["test_size"],
            stratify=class_labels
        )

        X_train = extractor.transform_train_data(X_train)
        X_test = extractor.transform_predict_data(X_test)
    elif config["extractor_cls"]=="FullSpectraExtractor":
        extractor = FullSpectraExtractor(**config["extractor_params"] or {})

        spectra, class_labels, spots = extractor.extract_train_data(TRAIN_DIR)

        X_train, X_test, y_train, y_test = train_test_split(
            spectra,
            class_labels,
            test_size=config["test_size"],
            stratify=class_labels
        )
    else:
        typer.echo(
        f"""extractor_cls must be in:
                        - PeakExtractor
                        - FullSpectraExtractor
                        """
        )
        raise typer.Exit(1)

    # Label encoding.
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Balancing training classes.
    if config["oversampling"]:
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # Select scaler.
    scalers = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
    }
    if config["scaler_cls"] is None:
        scaler_cls = None
    else:
        try:
            scaler_cls = scalers[config["scaler_cls"]]
        except KeyError:
            typer.echo(
                f"""scaler_cls parameter must be in:
                                - StandardScaler
                                - MinMaxScaler
                                - null
                            """
            )
            raise typer.Exit(1)

    # Select dimensionality reduction methond.
    dim_reducers = {
        "PCA": PCA,
        "SVD": TruncatedSVD
    }
    if config["dim_reducer_cls"] is None:
        dim_reducer_cls = None
    else:
        try:
            dim_reducer_cls = dim_reducers[config["dim_reducer_cls"]]
        except KeyError:
            typer.echo(
                f"""dim_reducer_cls parameter must be in:
                                - PCA
                                - SVD
                                - null
                            """
            )
            raise typer.Exit(1)

    # Check if further dimensionality reduction is possible
    n_features = X_train.shape[1]
    if dim_reducer_cls is not None and (n_features < int(config["n_components"])):
        typer.echo(
            f"""Feature length is smaller than n_components!
                Dimensionality can not be further reduced!
            """
        )
        raise typer.Exit(1)

    # Select classifier.
    classifiers = {
        "LogisticRegression": LogisticRegression,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "PLS-DA": PLSRegression,
        "OPLS-DA": None, # Separately defined
        "SVC": SVC,
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": XGBClassifier,
    }

    try:
        classifier_cls = classifiers[config["classifier_cls"]]
    except KeyError:
        typer.echo(
            f"""classifier_cls parameter must be in:
                         - LogisticRegression
                         - LinearDiscriminantAnalysis
                         - QuadraticDiscriminantAnalysis
                         - PLS-Da
                         - OPLS-Da
                         - SVC
                         - RandomForestClassifier
                         - XGBClassifier
                         """
        )
        raise typer.Exit(1)
    # Initialize pipeline.
    if config["classifier_cls"] == "OPLS-DA":
        # OPLS-DA pipeline.
        opls_da = SKLPipeline([
                ("opls", OPLS()),
                ("pls", PLSRegression())
        ])
        # Set classifier_params for OPLS-DA.
        opls_da.set_params(**config["classifier_params"] or {})
        pipeline = OneVsRestClassifier(opls_da)
    else:
        # All other model pipelines.
        pipeline = generate_pipeline(
            classifier_cls=classifier_cls,
            classifier_params=config["classifier_params"],
            scaler_cls=scaler_cls,
            dim_reducer_cls=dim_reducer_cls,
            n_components=int(config["n_components"]),
        )

    # Encoding y_train for PLS-DA.
    if config["classifier_cls"] == "PLS-DA":
        ohe = OneHotEncoder(sparse_output=False)
        y_train = ohe.fit_transform(y_train.reshape(-1, 1))

    # Training pipeline.
    pipeline.fit(X_train, y_train)

    # Compute test predictions.
    y_pred = pipeline.predict(X_test)

    # Transform y_train back from OHE for PLS-DA.
    if config["classifier_cls"] == "PLS-DA":
        y_pred = np.argmax(y_pred, axis=1)

    # Compute performance scores.
    performance_scores = {
        "ACC: ": accuracy_score(y_test, y_pred),
        "PREC: ": precision_score(y_test, y_pred, average="macro"),
        "REC: ": recall_score(y_test, y_pred, average="macro"),
        "F1: ": f1_score(y_test, y_pred, average="macro"),
        "CONF. MATRIX: \n": confusion_matrix(y_test, y_pred),
    }

    # Write pipeline scores achieved to file.
    timestamp = datetime.now()
    write_performance_scores(
        TRAINING_RESULTS_FILENAME,
        timestamp,
        config["classifier_cls"],
        performance_scores,
    )

    model_components = {
        "extractor": extractor,
        "pipeline": pipeline,
        "le": le
    }

    # Save model.
    joblib.dump(model_components, MODEL_SAVEFILE)

@app.command()
def predict():
    """
    Loads the extractor, pipeline model and label encoder previously created by
    using the 'train' command. Computes class predictions for the files inside
    'data_predict' and writes the labels and corresponding filenames to
    'cli_files/predictions.csv'.
    """
    # Check if cli_files directory exists.
    if not CLI_FILES_DIR.is_dir():
        typer.echo(
            f"""Could not find {CLI_FILES_DIR}!\n
                    Please, create the directory with the CLI files!"""
        )
        raise typer.Exit(1)

    # Check if model savefile exists.
    if not MODEL_SAVEFILE.exists():
        typer.echo(
            f"""Could not find {MODEL_SAVEFILE}!\n
                    Please, first use the 'train' command to
                    train a classifier pipeline!"""
        )
        raise typer.Exit(1)

    # Check if data_predict directory exists.
    if not PREDICT_DIR.is_dir():
        typer.echo(
            f"""Could not find {PREDICT_DIR}!\n
                    Please, create the directory with the prediction files!"""
        )
        raise typer.Exit(1)

    # Loads all model parts.
    model = joblib.load(MODEL_SAVEFILE)
    extractor = model["extractor"]
    pipeline = model["pipeline"]
    le = model["le"]

    # Extracting measurements.
    X, filenames = extractor.extract_predict_data(PREDICT_DIR)

    # Mapping peaks to consensus masses.
    if isinstance(extractor, PeakExtractor):
        X = extractor.transform_predict_data(X)

    # Compute predictions and decoded labels.
    predictions = pipeline.predict(X)

    # Transform y_train back from OHE for PLS-DA.
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # Generate readable prediction labels.
    pred_labels = le.inverse_transform(predictions)

    # Write predictions, labels, filepaths to .csv file.
    write_predictions(PREDICTIONS_FILENAME, pred_labels, filenames)

def main():
    """
    Runs Typer app.
    """
    app()

if __name__ == "__main__":
    """
    Entry point for the mtc CLI Typer application.
    """
    main()
