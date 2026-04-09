"""Utility helpers for the ML project.

This module provides helper functions for saving Python objects
and other common utilities used across the training and prediction pipelines.
"""

import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException

import joblib


def save_object(obj, file_path):
    """Serialize an object to disk using joblib.

    Parameters
    ----------
    obj : any
        The Python object to serialize.
    file_path : str
        Path to the target file where the object will be saved.

    Raises
    ------
    CustomException
        If saving fails due to an I/O or serialization error.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
