import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS Styling
# =========================

st.markdown("""
    <style>
    /* Hide Streamlit header and toolbar */
    header {
        visibility: hidden;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stDeployButton {
        visibility: hidden;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Title styling */
    .title-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }
    
    .subtitle {
        color: #475569;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Input section styling */
    .input-section {
        background: none;
        padding: 0;
        border-radius: 0;
        box-shadow: none;
        margin-bottom: 1.5rem;
        border: none;
    }
    
    .section-header {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #0891b2;
        padding-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #0891b2 0%, #0284c7 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.75rem 3rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 20px rgba(8, 145, 178, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(8, 145, 178, 0.7);
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    }
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        font-size: 1rem;
        background: #ffffff;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #0891b2;
        box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.15);
    }
    
    /* Select box styling */
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        font-size: 1rem;
        background: #ffffff;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #0891b2 0%, #0284c7 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(8, 145, 178, 0.3);
    }
    

    .stColumn {
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Load model and scaler
# =========================

with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# =========================
# Header
# =========================

st.markdown("""
    <div class="title-container">
        <h1 class="main-title">House Price Predictor</h1>
        
    """, unsafe_allow_html=True)


# =========================
# User Inputs
# =========================

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Location Details</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=-122.23, help="Geographic longitude coordinate")
with col2:
    latitude = st.number_input("Latitude", value=37.88, help="Geographic latitude coordinate")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Property Characteristics</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    housing_median_age = st.number_input("Housing Median Age", value=41, min_value=1, max_value=100, help="Median age of houses in the block")
with col2:
    total_rooms = st.number_input("Total Rooms", value=880, min_value=1, help="Total number of rooms in the block")
with col3:
    total_bedrooms = st.number_input("Total Bedrooms", value=129, min_value=1, help="Total number of bedrooms in the block")

col1, col2, col3 = st.columns(3)
with col1:
    population = st.number_input("Population", value=322, min_value=1, help="Total population in the block")
with col2:
    households = st.number_input("Households", value=126, min_value=1, help="Total number of households in the block")
with col3:
    median_income = st.number_input("Median Income", value=8.3252, min_value=0.0, help="Median income (in tens of thousands)")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header"> Ocean Proximity</h2>', unsafe_allow_html=True)

ocean_proximity = st.selectbox(
    "Select proximity to ocean",
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
    help="How close is the property to the ocean?"
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Feature Engineering
# (Must match training)
# =========================

total_rooms_log = np.log(total_rooms + 1)
total_bedrooms_log = np.log(total_bedrooms + 1)
population_log = np.log(population + 1)
households_log = np.log(households + 1)

bedroom_ratio = total_bedrooms_log / total_rooms_log
household_rooms = total_rooms_log / households_log

labels = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_encoded = [1 if ocean_proximity == label else 0 for label in labels]

# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame([[
    longitude,
    latitude,
    housing_median_age,
    total_rooms_log,
    total_bedrooms_log,
    population_log,
    households_log,
    median_income,
    ocean_encoded[0],
    ocean_encoded[1],
    ocean_encoded[2],
    ocean_encoded[3],
    ocean_encoded[4],
    bedroom_ratio,
    household_rooms
]], columns=[
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    '<1H OCEAN',
    'INLAND',
    'ISLAND',
    'NEAR BAY',
    'NEAR OCEAN',
    'bedroom_ratio',
    'household_rooms'
])

# =========================
# Prediction
# =========================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔮 Predict House Price"):
        with st.spinner('🤔 Analyzing property data...'):
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #0891b2 0%, #0284c7 100%);
                    color: white;
                    padding: 3rem 2rem;
                    border-radius: 20px;
                    text-align: center;
                    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                    margin: 2rem 0;
                    animation: slideIn 0.6s ease-out;
                ">
                    <h2 style="margin: 0; font-size: 1.5rem; font-weight: 600; opacity: 0.9;">Estimated Property Value</h2>
                    <h1 style="margin: 1rem 0; font-size: 3.5rem; font-weight: 800;">${prediction:,.2f}</h1>
                    
                </div>
            """, unsafe_allow_html=True)
            

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        
    </div>
    """, unsafe_allow_html=True)
