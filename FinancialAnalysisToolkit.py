import os
import numpy as np
import pandas as pd

from mom_trans.classical_strategies import (
    ModifiedMACDStrategy as NewMACDStrategy,
    calculate_modified_returns as calculate_modified_returns,
    calculate_modified_daily_vol as calculate_modified_daily_vol,
    calculate_modified_vol_scaled_returns as calculate_modified_vol_scaled_returns,
)

WINSORIZE_THRESHOLD = 5  # Threshold multiple for winsorization
HALFLIFE_WINSORIZE = 252


def load_changepoint_results(file_path: str, window_length: int) -> pd.DataFrame:
    """Load changepoint detection results from a CSV file into a DataFrame.
    Fills missing values by carrying forward the previous row's information for changepoint location and severity.

    Args:
        file_path (str): Path to the CSV file containing the results.
        window_length (int): Lookback window length, necessary for filling in the blanks for normalized location.

    Returns:
        pd.DataFrame: DataFrame with changepoint severity and location information.
    """
    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # Drop if first values are NA
        .assign(
            normalized_cp_location=lambda row: (row["t"] - row["cp_location"])
            / window_length
        )  # Fill by assigning the previous changepoint and score, then recalculate normalized location
    )


def prepare_changepoint_features(folder_path: str, window_length: int) -> pd.DataFrame:
    """Load changepoint detection results for all assets into a DataFrame.

    Args:
        folder_path (str): Folder path containing CSVs with the changepoint detection results.
        window_length (int): Lookback window length.

    Returns:
        pd.DataFrame: DataFrame with changepoint severity and location information for all assets.
    """
    return pd.concat(
        [
            load_changepoint_results(
                os.path.join(folder_path, file), window_length
            ).assign(ticker=os.path.splitext(file)[0])
            for file in os.listdir(folder_path)
        ]
    )


def generate_deep_momentum_features(asset_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare input features for a deep learning model.

    Args:
        asset_data (pd.DataFrame): Time-series data for an asset with a 'close' column.

    Returns:
        pd.DataFrame: Input features.
    """
    asset_data = asset_data[
        ~asset_data["close"].isna()
        | ~asset_data["close"].isnull()
        | (asset_data["close"] > 1e-8)  # Price is zero
    ].copy()

    # Winsorize using rolling 5X standard deviations to remove outliers
    asset_data["price"] = asset_data["close"]
    ewm = asset_data["price"].ewm(halflife=HALFLIFE_WINSORIZE)
    means = ewm.mean()
    stds = ewm.std()
    asset_data["price"] = np.minimum(
        asset_data["price"], means + WINSORIZE_THRESHOLD * stds
    )
    asset_data["price"] = np.maximum(
        asset_data["price"], means - WINSORIZE_THRESHOLD * stds
    )

    asset_data["daily_returns"] = calculate_modified_returns(asset_data["price"])
    asset_data["daily_vol"] = calculate_modified_daily_vol(asset_data["daily_returns"])

    # Volatility scaling and shift to be next day returns
    asset_data["target_returns"] = calculate_modified_vol_scaled_returns(
        asset_data["daily_returns"], asset_data["daily_vol"]
    ).shift(-1)

    def calculate_normalized_returns(day_offset):
        return (
            calculate_modified_returns(asset_data["price"], day_offset)
            / asset_data["daily_vol"]
            / np.sqrt(day_offset)
        )

    asset_data["norm_daily_return"] = calculate_normalized_returns(1)
    asset_data["norm_monthly_return"] = calculate_normalized_returns(21)
    asset_data["norm_quarterly_return"] = calculate_normalized_returns(63)
    asset_data["norm_biannual_return"] = calculate_normalized_returns(126)
    asset_data["norm_annual_return"] = calculate_normalized_returns(252)

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        asset_data[
            f"modified_macd_{short_window}_{long_window}"
        ] = NewMACDStrategy.calculate_signal(asset_data["price"], short_window, long_window)

    # Date features
    if len(asset_data):
        asset_data["day_of_week"] = asset_data.index.dayofweek
        asset_data["day_of_month"] = asset_data.index.day
        asset_data["week_of_year"] = asset_data.index.weekofyear
        asset_data["month_of_year"] = asset_data.index.month
        asset_data["year"] = asset_data.index.year
        asset_data["date"] = asset_data.index  # Duplication but sometimes makes life easier
    else:
        asset_data["day_of_week"] = []
        asset_data["day_of_month"] = []
        asset_data["week_of_year"] = []
        asset_data["month_of_year"] = []
        asset_data["year"] = []
        asset_data["date"] = []

    return asset_data.dropna()


def merge_changepoint_features(
    features: pd.DataFrame, changepoint_folder: pd.DataFrame, window_length: int
) -> pd.DataFrame:
    """Combine changepoint features with deep momentum features.

    Args:
        features (pd.DataFrame): Deep momentum features.
        changepoint_folder (pd.DataFrame): Folder containing changepoint detection results.
        window_length (int): Lookback window used for changepoint detection.

    Returns:
        pd.DataFrame: Features including changepoint detection score and location.
    """
    features = features.merge(
        prepare_changepoint_features(changepoint_folder, window_length)[
            ["ticker", "normalized_cp_location", "cp_score"]
        ]
        .rename(
            columns={
                "normalized_cp_location": f"normalized_cp_loc_{window_length}",
                "cp_score": f"cp_score_{window_length}",
            }
        )
        .reset_index(),  # For date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
