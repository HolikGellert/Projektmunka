import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FeatureEngineer
from prediction.config import PredictionConfig as Config

logger = logging.getLogger(__name__)


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    feature_cols: List[str]
    lookback: int
    scaler: StandardScaler


class PredictionDataPrep:
    """Handles loading, merging, feature building, and sequence creation."""

    def __init__(self):
        self.scaler: StandardScaler | None = None

    def load_raw(self) -> pd.DataFrame:
        """Reads raw energy + weather and merges on date."""
        logger.info("Loading energy data with compression=zip")
        energy = pd.read_csv(Config.RAW_ENERGY_FILE, compression="zip")
        energy["day"] = pd.to_datetime(energy["day"])

        logger.info("Loading weather data")
        weather = pd.read_csv(Config.RAW_WEATHER_FILE)
        weather["date"] = pd.to_datetime(weather["date"], format="%Y%m%d")

        merged = pd.merge(
            energy,
            weather,
            left_on="day",
            right_on="date",
            how="inner",
        ).drop(columns=["date"])

        merged = merged.dropna().sort_values(["LCLid", "day"])
        return merged

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds temporal/holiday features to align with existing style."""
        fe = FeatureEngineer(country_code="UK")
        enriched = fe.add_features(df)
        return enriched

    def _fit_scaler(self, df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
        self.scaler = StandardScaler()
        self.scaler.fit(df[feature_cols])
        return self.scaler

    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] | None = None,
        lookback: int | None = None,
        stride: int | None = None,
    ) -> SequenceData:
        """Converts dataframe to sliding windows suitable for RNN/TCN models."""
        feature_cols = feature_cols or Config.FEATURE_COLS
        lookback = lookback or Config.LOOKBACK
        stride = stride or Config.SEQUENCE_STRIDE

        df = df.sort_values(["LCLid", "day"]).reset_index(drop=True)
        self._fit_scaler(df, feature_cols)

        X_windows, y_targets = [], []
        groups = df.groupby("LCLid")

        for _, group in groups:
            group = group.reset_index(drop=True)
            features = self.scaler.transform(group[feature_cols])
            targets = group[Config.TARGET_COL].values

            for idx in range(lookback, len(group), stride):
                X_windows.append(features[idx - lookback : idx])
                y_targets.append(targets[idx])

        X_np = np.stack(X_windows)
        y_np = np.array(y_targets).reshape(-1, 1)

        # Down-sample to cap training volume
        if len(X_np) > Config.MAX_SEQUENCES:
            rng = np.random.default_rng(seed=Config.RANDOM_STATE)
            idx = rng.choice(len(X_np), size=Config.MAX_SEQUENCES, replace=False)
            X_np = X_np[idx]
            y_np = y_np[idx]

        return SequenceData(
            X=X_np,
            y=y_np,
            feature_cols=feature_cols,
            lookback=lookback,
            scaler=self.scaler,
        )

    def train_test_split(
        self, seq_data: SequenceData, test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simple chronological split for sequences."""
        cutoff = int(len(seq_data.X) * (1 - test_ratio))
        X_train, X_test = seq_data.X[:cutoff], seq_data.X[cutoff:]
        y_train, y_test = seq_data.y[:cutoff], seq_data.y[cutoff:]
        return X_train, X_test, y_train, y_test

    def train_val_test_split(
        self,
        seq_data: SequenceData,
        val_ratio: float = Config.VAL_RATIO,
        test_ratio: float = Config.TEST_RATIO,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Chronological split into train/val/test. Keeps ordering to prevent leakage.
        """
        n = len(seq_data.X)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_end = max(n - test_size - val_size, 0)
        val_end = max(n - test_size, train_end)

        X_train, y_train = seq_data.X[:train_end], seq_data.y[:train_end]
        X_val, y_val = seq_data.X[train_end:val_end], seq_data.y[train_end:val_end]
        X_test, y_test = seq_data.X[val_end:], seq_data.y[val_end:]

        return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_sequence_data(
    lookback: int | None = None,
    feature_cols: List[str] | None = None,
) -> SequenceData:
    """Convenience function used by notebooks/scripts."""
    prep = PredictionDataPrep()
    merged = prep.add_features(prep.load_raw())
    return prep.build_sequences(merged, feature_cols=feature_cols, lookback=lookback)
