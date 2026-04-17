import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Fertilizer Recommendation", page_icon="🌱", layout="wide")

# --- Custom CSS for aesthetic enhancement ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #2E7D32;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #E8F5E9;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #4CAF50;
    }
    .result-card h3 {
        color: #1B5E20 !important;
        margin-bottom: 10px;
    }
    .result-text {
        color: #1B5E20;
        font-size: 32px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('fertilizer_recommendation.csv')

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Get unique values for categorical features
soil_types = sorted(df['Soil_Type'].unique())
crop_types = sorted(df['Crop_Type'].unique())
growth_stages = sorted(df['Crop_Growth_Stage'].unique())
seasons = sorted(df['Season'].unique())
irrigation_types = sorted(df['Irrigation_Type'].unique())
previous_crops = sorted(df['Previous_Crop'].unique())
regions = sorted(df['Region'].unique())
fertilizers_used = sorted(df['Fertilizer_Used_Last_Season'].unique())

# --- Header ---
st.markdown('<p class="big-font">🌱 Intelligent Fertilizer Recommendation System</p>', unsafe_allow_html=True)
st.markdown("Enter your farm parameters below to receive a personalized, ML-driven fertilizer recommendation.")
st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🌍 Soil & Environment", "🌿 Crop Details", "🕰️ Farming History"])

# Dictionary to hold the inputs
inputs = {}

with tab1:
    st.header("Soil Characteristics & Weather")
    col1, col2 = st.columns(2)
    with col1:
        inputs['Soil_Type'] = st.selectbox("Soil Type", soil_types)
        inputs['Soil_pH'] = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        inputs['Soil_Moisture'] = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        inputs['Organic_Carbon'] = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col2:
        inputs['Electrical_Conductivity'] = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, value=1.5, step=0.1)
        inputs['Temperature'] = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        inputs['Humidity'] = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        inputs['Rainfall'] = st.number_input("Rainfall (mm)", min_value=0.0, value=1000.0, step=10.0)

with tab2:
    st.header("Crop Details")
    col1, col2 = st.columns(2)
    with col1:
        inputs['Crop_Type'] = st.selectbox("Crop Type", crop_types)
        inputs['Crop_Growth_Stage'] = st.selectbox("Growth Stage", growth_stages)
    with col2:
        inputs['Season'] = st.selectbox("Season", seasons)
        inputs['Region'] = st.selectbox("Geographical Region", regions)

with tab3:
    st.header("Historical Data & Nutrients")
    col1, col2 = st.columns(2)
    with col1:
        inputs['Irrigation_Type'] = st.selectbox("Irrigation Type", irrigation_types)
        inputs['Previous_Crop'] = st.selectbox("Previous Crop", previous_crops)
        inputs['Fertilizer_Used_Last_Season'] = st.selectbox("Fertilizer Used Last Season", fertilizers_used)
    with col2:
        inputs['Yield_Last_Season'] = st.number_input("Yield Last Season (tons/ha)", min_value=0.0, value=5.0, step=0.1)
        st.markdown("**Soil Nutrients (NPK)**")
        inputs['Nitrogen_Level'] = st.number_input("Nitrogen Level (N)", min_value=0, value=50)
        inputs['Phosphorus_Level'] = st.number_input("Phosphorus Level (P)", min_value=0, value=50)
        inputs['Potassium_Level'] = st.number_input("Potassium Level (K)", min_value=0, value=50)

st.markdown("---")

# Feature order EXACTLY as it appears in the training data:
feature_order = [
    'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level', 'Temperature', 'Humidity', 
    'Rainfall', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 
    'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season', 'Yield_Last_Season'
]

# --- Prediction Action ---
if st.button("🔮 Get Recommendation", use_container_width=True):
    # Assemble input dataframe
    input_list = [inputs[col] for col in feature_order]
    
    # Create pandas dataframe with single row
    input_df = pd.DataFrame([input_list], columns=feature_order)
    # Lowercase column names because the model was trained on lowercased columns
    input_df.columns = input_df.columns.str.lower()
    
    try:
        # The model is a Pipeline that contains the ColumnTransformer, so we pass raw inputs directly
        prediction = model.predict(input_df)
        
        st.markdown(f"""
        <div class="result-card">
            <h3>Recommended Fertilizer:</h3>
            <p class="result-text">{prediction[0]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed. Error details: {e}")
