import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import time
st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🍔", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #ffffff;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
st.title("🍔 Food Delivery Time Prediction")
st.markdown("Predict how long your food delivery will take based on delivery partner details and distance.")
