import os
import sys
import pytest

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import load_data, preprocess_data, load_model, train_model_if_needed

def get_absolute_path(relative_path):
    """Get the absolute path for a file relative to this script."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', relative_path))

@pytest.fixture
def sample_data_file():
    return get_absolute_path('data/housing.csv')

@pytest.fixture
def sample_model_file():
    return get_absolute_path('model/housing_price_model.joblib')

@pytest.fixture
def sample_save_path():
    return get_absolute_path('model/trained_housing_price_model.joblib')

def test_load_data(sample_data_file):
    """Test loading data from the CSV file."""
    data = load_data(sample_data_file)
    assert data is not None, "Data loading failed."
    assert len(data) > 0, "Loaded data is empty."

def test_preprocess_data(sample_data_file):
    """Test data preprocessing."""
    data = load_data(sample_data_file)
    X, y = preprocess_data(data)
    assert X is not None and y is not None, "Preprocessing failed."
    assert len(X) > 0 and len(y) > 0, "Preprocessed data is empty."

def test_load_model(sample_model_file):
    """Test loading the model."""
    model = load_model(sample_model_file)
    assert model is not None, "Model loading failed."

def test_train_model_if_needed(sample_data_file, sample_save_path):
    """
    Test the train_model_if_needed function.
    It verifies that the model is trained and saved correctly.
    """
    # Load and preprocess data
    data = load_data(sample_data_file)
    X, y = preprocess_data(data)
    
    # Train the model
    model = train_model_if_needed(X, y, sample_save_path)
    
    # Assertions
    assert model is not None, "Training failed."
    assert os.path.exists(sample_save_path), "Model save path does not exist."
