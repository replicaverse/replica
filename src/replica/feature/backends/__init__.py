"""Shared LM backend(s) used by multiple concepts, e.g. ``HuggingFaceLM``."""

from .huggingface import HuggingFaceLM

__all__ = ["HuggingFaceLM"]
