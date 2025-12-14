from pathlib import Path


class PredictionConfig:
    """Configuration for the deep-learning consumption predictor."""

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"

    # Reuse existing raw assets
    RAW_ENERGY_FILE = Path(__file__).parent.parent / "data" / "raw" / "daily_dataset.csv"
    RAW_WEATHER_FILE = Path(__file__).parent.parent / "data" / "raw" / "london_weather.csv"

    # Processed sequence store (ignored by git)
    PROCESSED_SEQUENCE_FILE = DATA_DIR / "merged_sequences.csv"

    # Sequence settings
    LOOKBACK = 21  # days
    SEQUENCE_STRIDE = 1  # finer stride for richer sequences
    MAX_SEQUENCES = None  # use full dataset when GPU is available
    TARGET_COL = "energy_sum"
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Feature order (keep in sync with scaler + inference)
    FEATURE_COLS = [
        "energy_sum",
        "energy_mean",
        "energy_max",
        "energy_std",
        "mean_temp",
        "max_temp",
        "min_temp",
        "global_radiation",
        "sunshine",
        "cloud_cover",
        "precipitation",
        "pressure",
        "is_weekend",
        "is_holiday",
        "day_of_week",
        "season",
    ]

    # Training defaults
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    MAX_TRAIN_BATCHES = None  # set to None to use full dataset
    MAX_EVAL_BATCHES = None   # set to None to use full eval
    EARLY_STOP_PATIENCE = 10  # stop if val loss does not improve
    GRAD_CLIP = 1.0

    # Artifact locations
    METADATA_PATH = MODEL_DIR / "metadata.json"
    SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
    LSTM_MODEL_PATH = MODEL_DIR / "lstm_regressor.pt"
    GRU_MODEL_PATH = MODEL_DIR / "gru_regressor.pt"
    CNN_MODEL_PATH = MODEL_DIR / "tcn_regressor.pt"

    RANDOM_STATE = 42
