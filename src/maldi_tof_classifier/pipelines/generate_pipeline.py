"""
Contains the function 'generate_pipeline' for creating a pipeline object for
classifying spectra, as well as Protocol objects for defining the interfaces
of the objects of the pipeline.

A pipeline may contain the following elements:
- Scaler: an instance of a data scaling model, e.g. StandardScaler. (Optional)
- DimReducer: an instance of a dimensionality reduction transformation model. (Optional)
- Classifier: a classification model instance.
"""

from typing import Protocol, Any
from sklearn.pipeline import Pipeline

class Classifier(Protocol):
    """
    Protocol for the classifier object used in the MALDI-TOF spectra
    classification pipeline.

    Must implement "fit" and "predict" following ScikitLearn conventions.
    """

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


class Scaler(Protocol):
    """
    Protocol for the scaler object (StandardScaler, MinMaxScaler) used in the MALDI-TOF
    spectra classification pipeline.

    Must implement "fit" and "transform" following ScikitLearn conventions.
    """

    def fit(self, X: Any, y: Any) -> Any: ...
    def transform(self, X: Any) -> Any: ...

class DimReducer(Protocol):
    """
    Protocol for the dimensionality reduction object (PCA, TruncatedSVD) used
    in the MALDI-TOF spectra classification pipeline.

    Must implement "fit" and "transform" following ScikitLearn conventions.
    """

    def fit(self, X: Any, y: Any) -> Any: ...
    def transform(self, X: Any) -> Any: ...


def generate_pipeline(
    classifier_cls: type[Classifier],
    classifier_params: dict | None = None,
    scaler_cls: type[Scaler] | None = None,
    dim_reducer_cls: type[DimReducer] | None = None,
    n_components: int | None = None,
) -> Pipeline:
    """
    Generates a pipeline for classifying MALDI-TOF spectra including
    optional data scaling and dimensionality reduction and a
    classifier model.

    Parameters
    -----------
    classifier_cls
        Instantiable class of the classifier used in the pipeline.
    classifier_params
        Parameters the classifier is initialized with.
    scaler_cls
        Instantiable class of the scaler (StandardScaler, MinMaxScaler) used in
        the pipeline.
    dim_reducer_cls
        Instantiable class of the dimensionality reduction transform to be
        used in the pipeline. 
    n_components
        Number of components parameter the DimReducer object is initialized
        with.

    Returns
    --------
    A pipeline object for classifying MALDI-TOF spectra.
    """
    classifier_params = classifier_params or {}

    if scaler_cls is None:
        scaler = "passthrough"
    else:
        scaler = scaler_cls()

    if dim_reducer_cls is None:
        dim_reducer = "passthrough"
    else:
        dim_reducer = dim_reducer_cls(n_components=n_components)

    classifier = classifier_cls(**classifier_params)

    return Pipeline(
        [
            ("scaler", scaler),
            ("dim_reducer", dim_reducer),
            ("classifier", classifier)
        ]
    )
