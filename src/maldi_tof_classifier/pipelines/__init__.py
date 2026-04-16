"""
Defines factory methods for creating MALDI-TOF MS measurement classfication
pipelines.

Modules:
- generate_pipeline: factory method for creating MALDI-TOF MS measurement
classfication pipeline containing optional scalind, dimensionality reduction
and a classifier.
"""

from .generate_pipeline import generate_pipeline

__all__ = ["generate_pipeline"]

