from pathlib import Path

class Config:
    # Random Seed
    RANDOM_STATE = 42
    
    # Clustering Settings
    N_CLUSTERS = 3
    
    # Paths (Dynamically resolves relative to the script location)
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    ENERGY_FILE = DATA_DIR / "daily_dataset.csv"
    WEATHER_FILE = DATA_DIR / "london_weather.csv"
    
    # Feature Groups
    # Features used for clustering (Normalized values usually)
    CLUSTER_FEATURES = [
        'energy_sum', 'energy_mean', 'energy_max', 'energy_std',
        'mean_temp', 'global_radiation', 'sunshine'
    ]