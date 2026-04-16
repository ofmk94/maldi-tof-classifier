"""
Defines factory methods for neural network based classification models for
spectral classification.

Modules:
- CNN1DClassifier: factory method for a 1D convolutional neural network based
  classifier.
- LSTMClassifier: factory method for a LSTM type recurrent neural network based
  classifier.
"""


from .CNN1DClassifier import CNN1D_Classifier
from .LSTMClassifier import LSTMClassifier

__all__ = ["CNN1DClassifier", "LSTMClassifier"]

