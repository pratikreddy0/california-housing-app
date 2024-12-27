import os
from app import load_data, preprocess_data, load_model, train_model_if_needed


def get_absolute_path(relative_path):
    """Get the absolute path for a file relative to this script."""
    return os.path.join(os.path.dirname(__file__), '..', relative_path)


def test_load_data():
    data_file = get_absolute_path('data/housing.csv')
    print(f"Loading data from {data_file} for testing...")
    data = load_data(data_file)
    assert data is not None, "Data loading failed."
    assert len(data) > 0, "Loaded data is empty."


def test_preprocess_data():
    data_file = get_absolute_path('data/housing.csv')
    print(f"Loading data from {data_file} for preprocessing...")
    data = load_data(data_file)
    assert data is not None, "Data loading failed during preprocessing."
    preprocessed_data = preprocess_data(data)
    assert preprocessed_data is not None, "Preprocessing failed."
    assert len(preprocessed_data) > 0, "Preprocessed data is empty."


def test_load_model():
    model_file = get_absolute_path('model/housing_price_model.joblib')
    print(f"Loading model from {model_file}...")
    model = load_model(model_file)
    assert model is not None, "Model loading failed."


def test_train_model_if_needed():
    data_file = get_absolute_path('data/housing.csv')
    print(f"Loading data from {data_file} for training...")
    data = load_data(data_file)
    assert data is not None, "Data loading failed during training."
    model = train_model_if_needed(data)
    assert model is not None, "Training failed."
