import pytest
import pandas as pd
from your_app import load_data, preprocess_data, train_model_if_needed
from sklearn.ensemble import RandomForestRegressor
import joblib

# Path to the dataset and model for testing
sample_data_file = 'path/to/your/housing.csv'
sample_save_path = 'path/to/your/trained_housing_price_model.joblib'

def test_load_data(sample_data_file):
    """Test loading of the data."""
    data = load_data(sample_data_file)
    assert isinstance(data, pd.DataFrame), "Data should be loaded as a DataFrame"
    assert 'median_house_value' in data.columns, "'median_house_value' column not found in data"

def test_preprocess_data(sample_data_file):
    """Test data preprocessing."""
    data = load_data(sample_data_file)
    X, feature_columns, le, imputer = preprocess_data(data)  # Now unpack only 4 values
    assert X.shape[0] == data.shape[0], "Number of rows should be the same after preprocessing"
    assert len(feature_columns) > 0, "Feature columns should not be empty"
    assert isinstance(le, LabelEncoder), "LabelEncoder object should be returned"
    assert isinstance(imputer, SimpleImputer), "SimpleImputer object should be returned"

def test_train_model_if_needed(sample_data_file, sample_save_path):
    """Test the train_model_if_needed function."""
    data = load_data(sample_data_file)
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Preprocess the data
    X_processed, feature_columns, le, imputer = preprocess_data(X)
    
    # Train model only if it doesn't exist
    model = train_model_if_needed(X_processed, y, sample_save_path)

    # Assert that the model is an instance of RandomForestRegressor
    assert isinstance(model, RandomForestRegressor), "Trained model should be a RandomForestRegressor"

    # Check if the model file was saved
    assert joblib.load(sample_save_path) is not None, "Model file was not saved"

def test_load_model(sample_save_path):
    """Test loading of the saved model."""
    model = joblib.load(sample_save_path)
    assert model is not None, "Model should be loaded successfully"
    assert isinstance(model, RandomForestRegressor), "Loaded model should be a RandomForestRegressor"
