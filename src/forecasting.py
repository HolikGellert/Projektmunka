import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report

class Forecaster:
    def __init__(self):
        self.regressor = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        self.classifier = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            n_jobs=-1, 
            random_state=42,
            class_weight='balanced'
        )

    def prepare_lag_features(self, df: pd.DataFrame, target='energy_sum', lags=[1, 2, 7]):
        """Creates previous days' consumption features for forecasting."""
        df = df.copy()
        df = df.sort_values(['LCLid', 'day'])
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('LCLid')[target].shift(lag)
            
        return df.dropna()

    def forecast_consumption(self, df: pd.DataFrame, feature_cols: list, target_col='energy_sum'):
        """Task: Short-term Energy Prediction (Regression)."""
        print(f"Training Consumption Forecaster on {len(df)} rows...")
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Time-based split (Last 20% is test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.regressor.fit(X_train, y_train)
        preds = self.regressor.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        return preds, y_test, mae

    def predict_cluster(self, df: pd.DataFrame, feature_cols: list, target_col='cluster'):
        """Task: Predict which cluster a day belongs to (Classification)."""
        print(f"Training Cluster Classifier...")
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classifier.fit(X_train, y_train)
        preds = self.classifier.predict(X_test)
        
        report = classification_report(y_test, preds)
        return preds, y_test, report