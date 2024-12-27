import os
import sys
import pytest

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import load_data, preprocess_data, load_model, train_model_if_needed


def get_absolute_path(relative_path):
    """Get the absolute path for a file relative to this script."""
    return os.path.join(os.path.dirname(__file__), '..', relative_path)


@pytest.fixture
def sample_data_file():
    return get_absolute_path('data/housing.csv')


@pytest.fixture
def sample_model_file():
    return get_absolute_path('model/housing_price_model.joblib')


def test_load_data(sample_data_file):
    data = load_data(sample_data_file)
    assert data is not None, "Data loading failed."
    assert len(data) > 0, "Loaded data is empty."


def test_preprocess_data(sample_data_file):
    data = load_data(sample_data_file)
    preprocessed_data = preprocess_data(data)
    assert preprocessed_data is not None, "Preprocessing failed."
    assert len(preprocessed_data) > 0, "Preprocessed data is empty."


def test_load_model(sample_model_file):
    model = load_model(sample_model_file)
    assert model is not None, "Model loading failed."


def test_train_model_if_needed(sample_data_file):
    data = load_data(sample_data_file)
    model = train_model_if_needed(data)
    assert model is not None, "Training failed."
