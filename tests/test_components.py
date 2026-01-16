import pytest
import pandas as pd
import numpy as np
import os
from src.config.loader import load_config
from src.data.loader import DataLoader
from src.features.preprocessing import Preprocessor

@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def sample_data():
    data = {
        'RowNumber': range(1, 11),
        'CustomerId': range(1001, 1011),
        'Surname': ['Smith'] * 10,
        'CreditScore': np.random.randint(300, 850, 10),
        'Geography': ['France', 'Spain'] * 5,
        'Gender': ['Female', 'Male'] * 5,
        'Age': np.random.randint(18, 90, 10),
        'Tenure': np.random.randint(0, 10, 10),
        'Balance': np.random.random(10) * 100000,
        'NumOfProducts': np.random.randint(1, 4, 10),
        'HasCrCard': np.random.randint(0, 2, 10),
        'IsActiveMember': np.random.randint(0, 2, 10),
        'EstimatedSalary': np.random.random(10) * 200000,
        'Exited': np.random.randint(0, 2, 10)
    }
    return pd.DataFrame(data)

def test_data_loader_clean(config, sample_data):
    loader = DataLoader(config)
    # Mock reading path not necessary as we test clean_data directly
    cleaned_df = loader.clean_data(sample_data.copy())
    
    # Check drops
    assert 'Surname' not in cleaned_df.columns
    # Check age filter (assuming config has 80)
    assert cleaned_df['Age'].max() <= loader.age_threshold

def test_preprocessor(config, sample_data):
    loader = DataLoader(config)
    cleaned_df = loader.clean_data(sample_data.copy())
    X_train, X_test, y_train, y_test = loader.split_data(cleaned_df)
    
    preprocessor = Preprocessor(config)
    X_train_processed, y_train_resampled, pipeline = preprocessor.fit_transform(X_train, y_train)
    
    assert X_train_processed.shape[0] == y_train_resampled.shape[0]
    assert pipeline is not None
    
    X_test_processed = preprocessor.transform(X_test, pipeline)
    assert X_test_processed.shape[0] == X_test.shape[0]
