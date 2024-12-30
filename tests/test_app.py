import pytest
import os
import tempfile
import shutil
import sys

# Add the directory of app.py to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app functions
from app import load_data, preprocess_data, train_model_if_needed, load_model  # Adjust as per your app structure

# Fixture for sample data file
@pytest.fixture
def sample_data_file():
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    temp_file.write('longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,median_house_value\n')
    temp_file.write('-122.23,37.88,41,880,129,322,626,8.3252,NEAR BAY,452600\n')
    temp_file.close()
    
    yield temp_file.name  # Return the file path for the test
    
    os.remove(temp_file.name)  # Clean up after the test

# Fixture for sample model save path
@pytest.fixture
def sample_save_path():
    temp_dir = tempfile.mkdtemp()
    yield os.path.join(temp_dir, 'housing_price_model.joblib')  # Return the model path
    
    shutil.rmtree(temp_dir)  # Recursively remove the directory and its contents

# Test function to load data
def test_load_data(sample_data_file):
    data = load_data(sample_data_file)
    assert data is not None, "Data loading failed"

# Test function to preprocess data
def test_preprocess_data(sample_data_file):
    data = load_data(sample_data_file)
    processed_data, _, _, _ = preprocess_data(data)
    assert processed_data is not None, "Data preprocessing failed"

# Test function to train the model if needed
def test_train_model_if_needed(sample_data_file, sample_save_path):
    data = load_data(sample_data_file)
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_processed, _, _, _ = preprocess_data(X)
    
    model = train_model_if_needed(X_processed, y, sample_save_path)
    
    # Ensure the model is returned and saved
    assert model is not None, "Model training failed"
    assert os.path.exists(sample_save_path), f"Model not saved at {sample_save_path}"

# Test function to load the model
def test_load_model(sample_save_path):
    # Ensure the file exists before testing
    if not os.path.exists(sample_save_path):
        pytest.skip(f"Model file not found at {sample_save_path}. Skipping test_load_model.")
    
    model = load_model(sample_save_path)
    assert model is not None, "Model loading failed"
