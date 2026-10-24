"""Evaluate models and benchmark caller-supplied datasets and predictors."""

from replica.analysis.evaluation.benchmark import benchmark
from replica.analysis.evaluation.evaluate import evaluate

__all__ = ["benchmark", "evaluate"]
