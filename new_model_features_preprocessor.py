# Model Inputs
import numpy as np
import pandas as pd
import datetime as dt
import enum
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing

# Type definitions
class DataTypes(enum.IntEnum):
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2

class InputTypes(enum.IntEnum):
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4
    TIME = 5

def get_single_col_by_input_type(input_type, column_definition):
    l = [tup[0] for tup in column_definition if tup[2] == input_type]
    if len(l) != 1:
        raise ValueError(f"Invalid number of columns for {input_type}")
    return l[0]

def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    return [tup[0] for tup in column_definition if tup[1] == data_type and tup[2] not in excluded_input_types]

class ModelFeatures:
    def __init__(
        self,
        df,
        total_time_steps,
        start_boundary=1990,
        test_boundary=2020,
        test_end=2021,
        changepoint_lbws=None,
        train_valid_sliding=False,
        transform_real_inputs=False,
        train_valid_ratio=0.9,
        split_tickers_individually=True,
        add_ticker_as_static=False,
        time_features=False,
        lags=None,
        asset_class_dictionary=None,
        static_ticker_type_feature=False,
    ):
        self._column_definition = [
            ("ticker", DataTypes.CATEGORICAL, InputTypes.ID),
            ("date", DataTypes.DATE, InputTypes.TIME),
            ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("norm_daily_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_monthly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_quarterly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_biannual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_annual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_8_24", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_16_48", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_32_96", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]
        df = df.dropna()
        df = df[df["year"] >= start_boundary].copy()
        years = df["year"]

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.total_time_steps = total_time_steps
        self.lags = lags

        if changepoint_lbws:
            for lbw in changepoint_lbws:
                self._column_definition.append(
                    (f"cp_score_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )
                self._column_definition.append(
                    (f"cp_rl_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )

        if time_features:
            self._column_definition.append(
                ("days_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_month", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("week_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )

            start_date = dt.datetime(start_boundary, 1, 1)
            days_from_start_max = (dt.datetime(test_end - 1, 12, 31) - start_date).days
            df["days_from_start"] = (df.index - start_date).days
            df["days_from_start"] = np.minimum(df["days_from_start"], days_from_start_max)

            df["days_from_start"] = MinMaxScaler().fit_transform(df[["days_from_start"]].values).flatten()
            df["day_of_week"] = MinMaxScaler().fit_transform(df[["day_of_week"]].values).flatten()
            df["day_of_month"] = MinMaxScaler().fit_transform(df[["day_of_month"]].values).flatten()
            df["week_of_year"] = MinMaxScaler().fit_transform(df[["week_of_year"]].values).flatten()

        if add_ticker_as_static:
            self._column_definition.append(
                ("static_ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
            )
            df["static_ticker"] = df["ticker"]
            if static_ticker_type_feature:
                df["static_ticker_type"] = df["ticker"].map(lambda t: asset_class_dictionary[t])
                self._column_definition.append(
                    ("static_ticker_type", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
                )

        self.transform_real_inputs = transform_real_inputs

        test = df.loc[years >= test_boundary]

        if split_tickers_individually:
            trainvalid = df.loc[years < test_boundary]
            if lags:
                tickers = (trainvalid.groupby("ticker")["ticker"].count() * (1.0 - train_valid_ratio)) >= total_time_steps
                tickers = tickers[tickers].index.tolist()
            else:
                tickers = list(trainvalid.ticker.unique())

            train, valid = [], []
            for ticker in tickers:
                calib_data = trainvalid[trainvalid.ticker == ticker]
                T = len(calib_data)
                train_valid_split = int(train_valid_ratio * T)
                train.append(calib_data.iloc[:train_valid_split, :].copy())
                valid.append(calib_data.iloc[train_valid_split:, :].copy())

            train = pd.concat(train)
            valid = pd.concat(valid)

            test = test[test.ticker.isin(tickers)]
        else:
            trainvalid = df.loc[years < test_boundary]
            dates = np.sort(trainvalid.index.unique())
            split_index = int(train_valid_ratio * len(dates))
            train_dates = pd.DataFrame({"date": dates[:split_index]})
            valid_dates = pd.DataFrame({"date": dates[split_index:]})

            train = (
                trainvalid.reset_index()
                .merge(train_dates, on="date")
                .set_index("date")
                .copy()
            )
            valid = (
                trainvalid.reset_index()
                .merge(valid_dates, on="date")
                .set_index("date")
                .copy()
            )
            if lags:
                tickers = (valid.groupby("ticker")["ticker"].count() > self.total_time_steps)
                tickers = tickers[tickers].index.tolist()
                train = train[train.ticker.isin(tickers)]
                valid = valid[valid.ticker.isin(tickers)]

        self._real_scalers = {}
        self._cat_scalers = {}
        self._target_scaler = None
        self._num_classes_per_cat_input = {}

        # Store min and max of each feature, used later for scaling
        self._column_maxes = {}
        self._column_mins = {}

        self.identifiers = train[get_single_col_by_input_type(InputTypes.ID, self._column_definition)].values
        self.identifiers_valid = valid[get_single_col_by_input_type(InputTypes.ID, self._column_definition)].values

        if self.lags:
            self._prepare_lags(train, valid)

        self._setup_scalers(train, valid, total_time_steps)
        self._add_static_inputs(train, valid)

        self._column_definition_df = pd.DataFrame(self._column_definition, columns=["feature", "type", "input_type"])
        self._real_inputs = extract_cols_from_data_type(DataTypes.REAL_VALUED, self._column_definition, [InputTypes.TIME])
        self._date_inputs = extract_cols_from_data_type(DataTypes.DATE, self._column_definition, [])
        self._cat_inputs = extract_cols_from_data_type(DataTypes.CATEGORICAL, self._column_definition, [InputTypes.TIME])

        self._data = {}
        self._data["train"] = self._preprocess_data(train)
        self._data["valid"] = self._preprocess_data(valid)

    def _prepare_lags(self, train, valid):
        for i in range(1, self.lags + 1):
            for col in ["target_returns"]:
                train[f"{col}_lag_{i}"] = train[col].shift(i)
                valid[f"{col}_lag_{i}"] = valid[col].shift(i)
                self._column_definition.append(
                    (f"{col}_lag_{i}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )

    def _setup_scalers(self, train, valid, total_time_steps):
        real_inputs = extract_cols_from_data_type(DataTypes.REAL_VALUED, self._column_definition, [InputTypes.TIME])
        real_inputs.remove("target_returns")
        if self.transform_real_inputs:
            self._real_scalers = {}
            for col in real_inputs:
                scaler = MinMaxScaler(feature_range=(0, 1))
                train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
                valid[col] = scaler.transform(valid[col].values.reshape(-1, 1))
                self._real_scalers[col] = scaler
                self._column_maxes[col] = scaler.data_max_[0]
                self._column_mins[col] = scaler.data_min_[0]

        self._target_scaler = MinMaxScaler(feature_range=(0, 1))
        train["target_returns"] = self._target_scaler.fit_transform(train["target_returns"].values.reshape(-1, 1))
        valid["target_returns"] = self._target_scaler.transform(valid["target_returns"].values.reshape(-1, 1))
        self._column_maxes["target_returns"] = self._target_scaler.data_max_[0]
        self._column_mins["target_returns"] = self._target_scaler.data_min_[0]

        # Identify categorical inputs and scale them
        categorical_inputs = extract_cols_from_data_type(DataTypes.CATEGORICAL, self._column_definition, [InputTypes.TIME])
        if categorical_inputs:
            self._cat_scalers = {}
            self._num_classes_per_cat_input = {}
            for col in categorical_inputs:
                le = sklearn.preprocessing.LabelEncoder()
                le.fit(train[col])
                train[col] = le.transform(train[col])
                valid[col] = le.transform(valid[col])
                self._cat_scalers[col] = le
                self._num_classes_per_cat_input[col] = len(le.classes_)

    def _add_static_inputs(self, train, valid):
        static_inputs = extract_cols_from_data_type(DataTypes.CATEGORICAL, self._column_definition, [InputTypes.STATIC_INPUT])
        for col in static_inputs:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(train[col])
            train[col] = le.transform(train[col])
            valid[col] = le.transform(valid[col])
            self._cat_scalers[col] = le
            self._num_classes_per_cat_input[col] = len(le.classes_)

    def _preprocess_data(self, data):
        if self.transform_real_inputs:
            data[self._real_inputs] = data[self._real_inputs].astype(np.float32)
        if self.lags:
            data[self._real_inputs + ["target_returns"]] = data[self._real_inputs + ["target_returns"]].astype(np.float32)
        if self.lags:
            data[self._cat_inputs] = data[self._cat_inputs].astype(np.longlong)
        else:
            data[self._cat_inputs] = data[self._cat_inputs].astype(np.longlong)
        data[self._date_inputs] = data[self._date_inputs].astype(np.float32)
        data[["days_from_start"]] = data[["days_from_start"]].astype(np.float32)
        if self.lags:
            return data[self._cat_inputs + self._real_inputs + ["target_returns"] + self._date_inputs + ["days_from_start", "days_from_start", "day_of_week", "day_of_month", "week_of_year"] + [f"{col}_lag_{i}" for i in range(1, self.lags + 1)]]
        return data[self._cat_inputs + self._real_inputs + ["target_returns"] + self._date_inputs + ["days_from_start", "days_from_start", "day_of_week", "day_of_month", "week_of_year"]]

    def get_dataloader(self, batch_size, data_type, shuffle=True):
        if data_type not in ["train", "valid"]:
            raise ValueError("Invalid data type. Use 'train' or 'valid'.")

        data = self._data[data_type].copy()
        data = data.dropna()

        if self.lags:
            dataset = TimeSeriesDataset(
                data=self._data[data_type],
                real_inputs=self._real_inputs,
                cat_inputs=self._cat_inputs,
                date_inputs=self._date_inputs,
                total_time_steps=self.total_time_steps,
                lags=self.lags,
                transform_real_inputs=self.transform_real_inputs,
            )
        else:
            dataset = TimeSeriesDataset(
                data=data,
                real_inputs=self._real_inputs,
                cat_inputs=self._cat_inputs,
                date_inputs=self._date_inputs,
                total_time_steps=self.total_time_steps,
                lags=self.lags,
                transform_real_inputs=self.transform_real_inputs,
            )

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        real_inputs,
        cat_inputs,
        date_inputs,
        total_time_steps,
        lags,
        transform_real_inputs,
    ):
        self._data = data
        self._real_inputs = real_inputs
        self._cat_inputs = cat_inputs
        self._date_inputs = date_inputs
        self._total_time_steps = total_time_steps
        self._lags = lags
        self._transform_real_inputs = transform
