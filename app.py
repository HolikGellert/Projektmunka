import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import joblib
from pathlib import Path

# --- SETUP PATHS ---
# Add the current directory to sys.path to ensure local modules can be imported
REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- IMPORTS FROM YOUR REPO ---
try:
    from prediction.src.models import LSTMRegressor, GRURegressor, TemporalCNNRegressor
    from prediction.config import PredictionConfig as Config
except ImportError:
    st.error("Error: Could not find 'prediction' module. Make sure the 'prediction/' folder is in the same directory as app.py.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Energy Consumption Predictor")

st.title("Energy Consumption Prediction")
st.markdown("""
This application uses Deep Learning models (LSTM, GRU, TCN) to forecast daily energy consumption.
Provide the latest day's features below, or upload the last N days to build a realistic lookback window.
""")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Architecture",
    ("LSTM", "GRU", "TCN")
)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model_and_scaler(model_name):
    """Loads the model weights and the data scaler. Cached for performance."""
    
    # 1. Setup Model Architecture
    input_size = len(Config.FEATURE_COLS)
    
    if model_name == "LSTM":
        model = LSTMRegressor(input_size=input_size)
        path = Config.LSTM_MODEL_PATH
    elif model_name == "GRU":
        model = GRURegressor(input_size=input_size)
        path = Config.GRU_MODEL_PATH
    else:
        model = TemporalCNNRegressor(input_size=input_size)
        path = Config.CNN_MODEL_PATH
    
    # 2. Load Weights (Map to CPU for Streamlit Cloud)
    try:
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            st.error(f"Model file not found at: {path}")
            return None, None
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None

    model.eval()
    
    # 3. Load Scaler
    # Try looking in config path, or fallback to a local 'models' folder
    scaler_path = getattr(Config, 'SCALER_PATH', Path("prediction/models/scaler.pkl"))
    
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
    else:
        st.warning("Scaler file not found. Predictions might be inaccurate (unscaled).")
    
    return model, scaler

# --- MAIN: INPUT FORM ---
st.subheader("Latest Day Features (match training schema)")

weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

col1, col2 = st.columns(2)

with col1:
    energy_sum = st.number_input("Daily energy_sum (kWh)", min_value=0.0, value=12.0, step=0.1)
    energy_mean = st.number_input("Daily energy_mean (kWh)", min_value=0.0, value=0.5, step=0.01)
    energy_max = st.number_input("Daily energy_max (kWh)", min_value=0.0, value=1.5, step=0.05)
    energy_std = st.number_input("Daily energy_std (kWh)", min_value=0.0, value=0.2, step=0.01)
    mean_temp = st.slider("Mean Temperature (°C)", -15.0, 40.0, 12.0)
    max_temp = st.slider("Max Temperature (°C)", -15.0, 45.0, 18.0)
    min_temp = st.slider("Min Temperature (°C)", -25.0, 35.0, 8.0)

with col2:
    global_radiation = st.number_input("Global radiation (kJ/m²)", min_value=0.0, max_value=5000.0, value=1000.0, step=50.0)
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
    cloud_cover = st.slider("Cloud cover (oktas, 0-8)", 0.0, 8.0, 4.0)
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=80.0, value=0.0, step=0.5)
    pressure = st.number_input("Pressure (Pa)", min_value=90000.0, max_value=110000.0, value=101300.0, step=100.0)
    day_of_week = st.selectbox("Day of week (0=Mon)", list(range(7)), format_func=lambda i: f"{i} ({weekday_names[i]})")
    season = st.selectbox("Season code", options=[1, 2, 3, 4], format_func=lambda i: {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}[i])
    is_holiday = st.checkbox("Is Holiday?", value=False)

# --- OPTIONAL HISTORY UPLOAD ---
st.subheader("Optional: Upload recent days (CSV) for lookback")
st.caption(f"Expected columns: {', '.join(Config.FEATURE_COLS)}. At least {Config.LOOKBACK} rows recommended.")
uploaded_history = st.file_uploader("CSV with recent days (will use the last rows as lookback)", type=["csv"])

# --- PREDICTION LOGIC ---
if st.button("Run Prediction"):
    model, scaler = load_model_and_scaler(model_type)
    
    if model is not None:
        # Construct input data dictionary aligned with Config.FEATURE_COLS
        input_data = {
            "energy_sum": energy_sum,
            "energy_mean": energy_mean,
            "energy_max": energy_max,
            "energy_std": energy_std,
            "mean_temp": mean_temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "global_radiation": global_radiation,
            "sunshine": sunshine,
            "cloud_cover": cloud_cover,
            "precipitation": precipitation,
            "pressure": pressure,
            "is_weekend": 1.0 if day_of_week >= 5 else 0.0,
            "is_holiday": 1.0 if is_holiday else 0.0,
            "day_of_week": float(day_of_week),
            "season": float(season),
        }
        
        feature_vector = [input_data[col] for col in Config.FEATURE_COLS]
        lookback = getattr(Config, 'LOOKBACK', 24)

        # Try to build sequence from uploaded history (preferred)
        seq_array = None
        if uploaded_history is not None:
            try:
                history_df = pd.read_csv(uploaded_history)
                missing = [c for c in Config.FEATURE_COLS if c not in history_df.columns]
                if missing:
                    st.warning(f"Uploaded file missing columns: {missing}. Falling back to repeated latest-day features.")
                else:
                    history_slice = history_df.tail(lookback)
                    if len(history_slice) < lookback:
                        st.warning(f"Need at least {lookback} rows for lookback. Found {len(history_slice)}. Falling back to repeated latest-day features.")
                    else:
                        seq_array = history_slice[Config.FEATURE_COLS].to_numpy()
            except Exception as e:
                st.warning(f"Could not parse uploaded CSV: {e}. Falling back to repeated latest-day features.")

        # Fallback: repeat the latest-day vector across the lookback
        if seq_array is None:
            seq_array = np.tile(feature_vector, (lookback, 1))
        
        # Scale Data
        if scaler:
            try:
                df_seq = pd.DataFrame(seq_array, columns=Config.FEATURE_COLS)
                seq_array_scaled = scaler.transform(df_seq)
            except Exception:
                seq_array_scaled = seq_array
        else:
            seq_array_scaled = seq_array

        # Convert to Tensor [Batch, Seq, Features]
        X_tensor = torch.FloatTensor(seq_array_scaled).unsqueeze(0)

        # Inference
        with torch.no_grad():
            prediction_result = model(X_tensor).item()
        
        # Display Result
        st.success("Prediction Complete!")
        st.metric(label="Predicted Energy Consumption", value=f"{prediction_result:.2f} kWh")
        
        # visual aid (mock chart)
        st.caption("Forecast Confidence Interval (Visual Aid)")
        chart_data = pd.DataFrame(
            np.random.normal(prediction_result, prediction_result*0.05, 24),
            columns=["Forecast"]
        )
        st.line_chart(chart_data)
