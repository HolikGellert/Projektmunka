import pandas as pd
import logging
from .config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.energy_path = Config.ENERGY_FILE
        self.weather_path = Config.WEATHER_FILE

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads energy and weather data, cleans, and merges them."""
        
        # 1. Load Energy Data
        logger.info(f"Loading energy data from {self.energy_path}")
        if not self.energy_path.exists():
            raise FileNotFoundError(f"File not found: {self.energy_path}")
        
        energy_df = pd.read_csv(self.energy_path)
        energy_df['day'] = pd.to_datetime(energy_df['day'])
        
        # Filter: Only keep days with full measurements (count >= 48)
        initial_rows = len(energy_df)
        energy_df = energy_df[energy_df['energy_count'] >= 48].copy()
        logger.info(f"Filtered incomplete days. Rows: {initial_rows} -> {len(energy_df)}")

        # 2. Load Weather Data
        logger.info(f"Loading weather data from {self.weather_path}")
        if not self.weather_path.exists():
            raise FileNotFoundError(f"File not found: {self.weather_path}")
            
        weather_df = pd.read_csv(self.weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y%m%d')
        
        # Handle specific missing data (Imputation based on prior knowledge)
        # 2012-03-02 cloud_cover imputation
        mask = weather_df['date'] == pd.to_datetime('2012-03-02')
        weather_df.loc[mask, 'cloud_cover'] = 6.0
        
        # 3. Merge
        logger.info("Merging datasets...")
        merged_df = pd.merge(
            energy_df, 
            weather_df, 
            left_on='day', 
            right_on='date', 
            how='inner'
        )
        merged_df.drop(columns=['date'], inplace=True)
        
        # Final cleanup
        merged_df.dropna(inplace=True)
        
        return merged_df