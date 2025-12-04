import pandas as pd
import holidays
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class FeatureEngineer:
    def __init__(self, country_code='UK'):
        self.uk_holidays = holidays.country_holidays(country_code)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds temporal features and holiday flags."""
        df = df.copy()
        
        # Date components
        df['year'] = df['day'].dt.year
        df['month'] = df['day'].dt.month
        df['day_of_week'] = df['day'].dt.dayofweek
        df['week'] = df['day'].dt.isocalendar().week
        
        # Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
        df['season'] = df['month'] % 12 // 3 + 1
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df['season_name'] = df['season'].map(season_map)
        
        # Weekend Flag
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Holiday Flag (Data Augmentation)
        # Using .get() is faster than list comprehension for large series
        df['is_holiday'] = df['day'].apply(lambda x: 1 if x in self.uk_holidays else 0)
        
        return df

    def scale_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
        """Standardizes features for clustering."""
        scaler = StandardScaler()
        scaled_data = df.copy()
        
        # Create new columns with suffix '_scaled'
        scaled_values = scaler.fit_transform(df[features])
        scaled_cols = [f"{col}_scaled" for col in features]
        
        scaled_df = pd.DataFrame(scaled_values, columns=scaled_cols, index=df.index)
        
        return pd.concat([scaled_data, scaled_df], axis=1), scaler