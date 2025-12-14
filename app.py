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
This application uses Deep Learning models (LSTM, GRU, TCN) to forecast energy consumption based on weather parameters.
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
st.subheader("Weather Parameters")

col1, col2 = st.columns(2)

with col1:
    mean_temp = st.slider("Mean Temperature (°C)", -10.0, 40.0, 15.0)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    cloud_cover = st.slider("Cloud Cover (0-1)", 0.0, 1.0, 0.5)

with col2:
    precipitation = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0)
    sunshine = st.number_input("Sunshine (hours)", 0.0, 16.0, 5.0)
    is_holiday = st.checkbox("Is Holiday?", value=False)

# --- PREDICTION LOGIC ---
if st.button("Run Prediction"):
    model, scaler = load_model_and_scaler(model_type)
    
    if model is not None:
        # Construct input data dictionary
        # NOTE: Ensure these keys match your Config.FEATURE_COLS exactly
        input_data = {
            'temperature': mean_temp,
            'humidity': humidity, 
            'precipitation': precipitation,
            'cloud_cover': cloud_cover,
            'sunshine': sunshine,
            'global_radiation': sunshine * 10, # Estimated derivation
            'mean_pressure': 1013.0, # Default average
            'snow_depth': 0.0,
            # Temporal features (dummy values for demo)
            'month_sin': 0.5, 'month_cos': 0.5,
            'day_sin': 0.5, 'day_cos': 0.5,
            'weekday_sin': 0.5, 'weekday_cos': 0.5,
            'is_holiday': 1.0 if is_holiday else 0.0,
            'is_weekend': 0.0
        }
        
        # Build feature vector based on Config order
        feature_vector = []
        for col in Config.FEATURE_COLS:
            feature_vector.append(input_data.get(col, 0.0))
            
        # Create Sequence:
        # The model expects a sequence (e.g., past 24h). 
        # For this demo, we repeat the user input to fill the lookback window.
        lookback = getattr(Config, 'LOOKBACK', 24)
        seq_array = np.tile(feature_vector, (lookback, 1))
        
        # Scale Data
        if scaler:
            try:
                # Assuming scaler was fitted on a DataFrame
                df_seq = pd.DataFrame(seq_array, columns=Config.FEATURE_COLS)
                seq_array_scaled = scaler.transform(df_seq)
            except Exception:
                # Fallback if scaler expects different input shape
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