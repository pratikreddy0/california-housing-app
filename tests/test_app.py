import sys
import os
import pytest

# Add the root directory (F:\california_housing_app) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from app.py
from app import load_data, preprocess_data, load_model, train_model_if_needed

# Test for loading data
def test_load_data():
    data = load_data('data/housing.csv')
    assert data is not None, "Data loading failed."
    assert len(data) > 0, "Loaded data is empty."

# Test for preprocessing data
def test_preprocess_data():
    data = load_data('data/housing.csv')
    processed_data, feature_columns, le, imputer = preprocess_data(data)
    assert processed_data is not None, "Preprocessing failed."
    assert len(processed_data) > 0, "Processed data is empty."
    assert len(processed_data[0]) == len(feature_columns), "Feature columns mismatch."

# Test for loading a model
def test_load_model():
    model = load_model('model/housing_price_model.joblib')
    assert model is not None, "Model loading failed."
    assert hasattr(model, "predict"), "Loaded model does not have a predict method."

# Test for training the model if needed
def test_train_model_if_needed():
    data = load_data('data/housing.csv')
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    model = train_model_if_needed(X, y, 'model/housing_price_model.joblib')
    
    assert model is not None, "Model training failed."
    assert hasattr(model, "predict"), "Trained model does not have a predict method."
