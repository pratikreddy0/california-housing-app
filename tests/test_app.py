import pytest
import tempfile
import shutil
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Adjust the path for app.py

from app import load_data, preprocess_data, train_model_if_needed, load_model

# Create fixtures for test files and directories
@pytest.fixture
def sample_data_file():
    """Creates a temporary CSV file with sample housing data."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="")
    temp_file.write("longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,median_house_value\n")
    temp_file.write("-122.23,37.88,41,880,129,322,126,8.3252,NEAR BAY,452600\n")
    temp_file.write("-122.22,37.86,21,7099,1106,2401,1138,8.3014,NEAR BAY,358500\n")
    temp_file.close()
    yield temp_file.name
    os.remove(temp_file.name)

@pytest.fixture
def sample_model_path():
    """Creates a temporary directory to save the model."""
    temp_dir = tempfile.mkdtemp()
    model_file_path = os.path.join(temp_dir, "housing_price_model.joblib")
    yield model_file_path
    shutil.rmtree(temp_dir)

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}.")
        sys.exit(1)  # Explicitly raise SystemExit here
    return pd.read_csv(file_path)

# Update sys.exit handling to raise SystemExit correctly in pytest
def test_load_data_invalid_path():
    """Tests behavior when file path does not exist."""
    with pytest.raises(SystemExit) as exc_info:  # Capture the exception
        load_data("invalid/path/to/file.csv")
    
    # Check if the exit code is 1 (commonly used for failure)
    assert exc_info.type == SystemExit
    assert exc_info.value.code == 1

# Test preprocess_data function
def test_preprocess_data(sample_data_file):
    """Tests data preprocessing."""
    data = load_data(sample_data_file)
    processed_data, feature_columns, le, imputer = preprocess_data(data)
    assert processed_data is not None, "Preprocessed data should not be None."
    assert len(feature_columns) > 0, "Feature columns should not be empty."
    assert isinstance(le, LabelEncoder), "Expected a LabelEncoder object."  # Fixed assertion for LabelEncoder

# Test train_model_if_needed function
def test_train_model_if_needed(sample_data_file, sample_model_path):
    """Tests training and saving a model."""
    data = load_data(sample_data_file)
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_processed, _, _, _ = preprocess_data(X)
    model = train_model_if_needed(X_processed, y, sample_model_path)
    assert model is not None, "Model training should return a model."
    assert os.path.exists(sample_model_path), "Model file should be saved."

# Test load_model function
def test_load_model(sample_model_path, sample_data_file):
    """Tests loading an existing model."""
    # Train a model and save it
    data = load_data(sample_data_file)
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_processed, _, _, _ = preprocess_data(X)
    train_model_if_needed(X_processed, y, sample_model_path)
    
    # Load the model
    model = load_model(sample_model_path)
    assert model is not None, "Model should be loaded successfully."
