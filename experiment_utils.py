import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
import shutil
import gc
import copy

import json



from settings.default import BACKTEST_AVERAGE_BASIS_POINTS

from settings.hp_grid import HP_MINIBATCH_SIZE

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def get_results_directory_name(
    experiment_name: str, training_interval: Tuple[int, int, int] = None
) -> str:
    """The directory name for saving results

    Args:
        experiment_name (str): name of experiment
        training_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.

    Returns:
        str: folder name
    """
    if training_interval:
        return os.path.join(
            "results", experiment_name, f"{training_interval[1]}-{training_interval[2]}"
        )
    else:
        return os.path.join(
            "results",
            experiment_name,
        )


def basis_point_suffix(basis_points: float = None) -> str:
    """Basis points suffix

    Args:
        basis_points (float, optional): bps value. Defaults to None.

    Returns:
        str: suffix name
    """
    if not basis_points:
        return ""
    return "_" + str(basis_points).replace(".", "_") + "_bps"


def interval_suffix(
    training_interval: Tuple[int, int, int], basis_points: float = None
) -> str:
    """Interval points suffix

    Args:
        training_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.
        basis_points (float, optional): bps value. Defaults to None.

    Returns:
        str: suffix name
    """
    return f"_{training_interval[1]}_{training_interval[2]}" + basis_point_suffix(
        basis_points
    )
