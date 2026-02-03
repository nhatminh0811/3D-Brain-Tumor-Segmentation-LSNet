"""Compatibility shim: keep `from model import get_model` working.

This module forwards to `src.models` package. By default it returns the
baseline SegResNet implementation so existing scripts keep working.
"""
from .models import get_model as _pkg_get_model


def get_model():
    return _pkg_get_model(name="baseline")
