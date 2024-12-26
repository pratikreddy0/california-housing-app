import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib  # For saving and loading models
import os  # To check file paths
import logging  # For logging actions

# Set file paths 
base_dir = os.path.dirname(__file__)  # Gets the directory of the current file
data_path = os.path.join(base_dir, "data", "housing.csv")
model_path = os.path.join(base_dir, "model", "housing_price_model.joblib")
metrics_path = os.path.join(base_dir, "model_metrics", "model_metrics.txt")
log_path = os.path.join(base_dir, "logs", "model_training.log")

# Page configuration
st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "This app predicts housing prices using a machine learning model trained on California housing data."
    }
)

# App title and description
st.title("California Housing Price Prediction App 🏡")
st.markdown("""
    ### 🏘️ Welcome to the California Housing Price Prediction App
    Use this app to explore data, train the model, and predict housing prices interactively!
""")

# Set up logging
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log the start of the application
logging.info("Streamlit application started.")

# Apply custom theme
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f4f7;
        }
        .sidebar .sidebar-content {
            background-color: #2e3b4e;
            color: white;
        }
        .stTitle {
            font-size: 3rem;
            color: #007f99;
        }
        .stText, .stMark {
            font-size: 1.2rem;
            color: #333;
        }
        .stButton button {
            background-color: #4caf50;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 1.1rem;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data(filepath):
    if os.path.exists(filepath):
        logging.info(f"Dataset loaded successfully from {filepath}.")
        return pd.read_csv(filepath)
    else:
        logging.error(f"Data file not found at {filepath}.")
        st.error(f"Data file not found at {filepath}")
        st.stop()

# Preprocess dataset
def preprocess_data(df):
    le = LabelEncoder()
    imputer = SimpleImputer(strategy="mean")
    
    df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])
    df = pd.get_dummies(df)
    feature_columns = list(df.columns)
    
    df = imputer.fit_transform(df)
    logging.info("Data preprocessing completed successfully.")
    return df, feature_columns, le, imputer

# Train and save the model only if it doesn't exist
def train_model_if_needed(X, y, save_path):
    if not os.path.exists(save_path):  # Check if model already exists
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, save_path)  # Save the trained model
        logging.info(f"Model trained and saved at {save_path}.")
        return model
    else:
        logging.info(f"Model already exists at {save_path}, skipping training.")
        return load_model(save_path)

# Load the saved model with caching
@st.cache_resource
def load_model(filepath):
    if os.path.exists(filepath):
        logging.info(f"Model loaded successfully from {filepath}.")
        return joblib.load(filepath)
    else:
        logging.error(f"Model file not found at {filepath}.")
        st.error(f"Model file not found at {filepath}")
        st.stop()

# Save metrics to file
def save_metrics(filepath, rmse, ci):
    with open(filepath, "w") as file:
        file.write(f"RMSE: {rmse:.2f}\n")
        file.write(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
    logging.info(f"Model metrics saved to {filepath}: RMSE={rmse:.2f}, CI=({ci[0]:.2f}, {ci[1]:.2f}).")

# Sidebar inputs
st.sidebar.header("Input Parameters")
longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("Housing Median Age", min_value=1, max_value=100, value=30)
total_rooms = st.sidebar.slider("Total Rooms", min_value=1, max_value=10000, value=500)
total_bedrooms = st.sidebar.slider("Total Bedrooms", min_value=1, max_value=5000, value=300)
population = st.sidebar.slider("Population", min_value=1, max_value=20000, value=1000)
households = st.sidebar.slider("Households", min_value=1, max_value=5000, value=400)
median_income = st.sidebar.slider("Median Income", min_value=0.0, max_value=15.0, value=3.5)
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", ['NEAR BAY', 'NEAR OCEAN', 'ISLAND', 'INLAND'])

# Display Correlation Matrix
if st.checkbox("Show Correlation Matrix"):
    housing = load_data(data_path)
    numeric_data = housing.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=20)
    st.pyplot(plt)
    logging.info("Correlation matrix displayed.")

# Train and predict
if st.sidebar.button("🚂 Train and Predict"):
    housing = load_data(data_path)
    X = housing.drop("median_house_value", axis=1)
    y = housing["median_house_value"]

    # Preprocess the data
    X_processed, feature_columns, le, imputer = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train model only if it doesn't exist
    model = train_model_if_needed(X_train, y_train, model_path)

    # Calculate RMSE and Confidence Interval
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    ci = stats.t.interval(0.95, len(predictions)-1, loc=rmse, scale=stats.sem(predictions))
    save_metrics(metrics_path, rmse, ci)

    st.success(f"Final RMSE: {rmse:.2f}")
    st.info(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
    logging.info(f"RMSE calculated: {rmse:.2f}, Confidence Interval: {ci}.")

    # Predict user input
    user_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # Encode ocean proximity
    user_data['ocean_proximity'] = le.transform(user_data['ocean_proximity'])
    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=feature_columns, fill_value=0)
    user_data = imputer.transform(user_data)

    # Load model and predict
    prediction = model.predict(user_data)
    st.success(f"Predicted Median House Value: ${prediction[0]:,.2f}")
    logging.info(f"Prediction made for user input: {user_data}, Predicted Value: ${prediction[0]:,.2f}")
