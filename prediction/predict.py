import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch

from prediction.config import PredictionConfig as Config
from prediction.src.models import GRURegressor, LSTMRegressor, TemporalCNNRegressor


MODEL_REGISTRY = {
    "lstm": (Config.LSTM_MODEL_PATH, LSTMRegressor),
    "gru": (Config.GRU_MODEL_PATH, GRURegressor),
    "tcn": (Config.CNN_MODEL_PATH, TemporalCNNRegressor),
}


def load_metadata(path: Path = Config.METADATA_PATH) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_cols": Config.FEATURE_COLS, "lookback": Config.LOOKBACK}


def prompt_sequence(feature_cols: List[str], lookback: int) -> np.ndarray:
    print(
        f"Enter the last {lookback} days of data.\n"
        f"Order for each day: {', '.join(feature_cols)}"
    )
    rows = []
    for i in range(lookback):
        while True:
            raw = input(f"Day -{lookback - i} values (comma separated): ").strip()
            try:
                values = [float(x.strip()) for x in raw.split(",")]
                if len(values) != len(feature_cols):
                    raise ValueError
                rows.append(values)
                break
            except ValueError:
                print(f"Please provide exactly {len(feature_cols)} numeric values separated by commas.")
    return np.array(rows)


def build_model(model_key: str, input_size: int):
    _, cls = MODEL_REGISTRY[model_key]
    if model_key == "tcn":
        return cls(input_size=input_size)
    return cls(input_size=input_size)


def main():
    parser = argparse.ArgumentParser(description="Predict next-day electricity consumption from weather + recent usage.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="lstm",
        help="Which trained model artifact to load.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit model path (overrides default).",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    feature_cols = metadata.get("feature_cols", Config.FEATURE_COLS)
    lookback = int(metadata.get("lookback", Config.LOOKBACK))

    model_path = args.model_path or MODEL_REGISTRY[args.model][0]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the models in "
            f"prediction/notebooks/prediction_workflow.ipynb first."
        )

    scaler = None
    if Config.SCALER_PATH.exists():
        scaler = joblib.load(Config.SCALER_PATH)
    else:
        print("Warning: scaler not found, using raw values. Predictions may be inaccurate.")

    user_sequence = prompt_sequence(feature_cols, lookback)
    if scaler:
        user_sequence = scaler.transform(user_sequence)

    model = build_model(args.model, input_size=len(feature_cols))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        tensor_input = torch.tensor(user_sequence, dtype=torch.float32).unsqueeze(0)
        prediction = model(tensor_input).item()

    print(f"Predicted next-day energy_sum: {prediction:.3f} kWh (approx.)")


if __name__ == "__main__":
    main()
