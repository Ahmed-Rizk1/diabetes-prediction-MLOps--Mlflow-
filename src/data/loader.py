import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging
import os

# Initialize logger
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_data_path = config['paths']['raw_data']
        self.drop_columns = config['data']['drop_columns']
        self.age_threshold = config['preprocessing']['age_threshold']
        self.target = config['data']['target']
        self.test_size = config['data']['test_size']
        self.random_state = config['data']['random_state']
        self.stratify = config['data']['stratify']

    def load_data(self) -> pd.DataFrame:
        """Load data from csv file."""
        if not os.path.exists(self.raw_data_path):
             # Try absolute path if relative fails, or check if it is relative to project root
             # Assuming running from root
             if not os.path.exists(self.raw_data_path):
                 raise FileNotFoundError(f"Data file not found at {self.raw_data_path}")
        
        logger.info(f"Loading data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning filters."""
        initial_shape = df.shape
        
        # Drop unnecessary columns
        df.drop(columns=self.drop_columns, axis=1, inplace=True, errors='ignore')
        
        # Filter by age
        df = df[df['Age'] <= self.age_threshold]
        
        logger.info(f"Data cleaned. Rows dropped: {initial_shape[0] - df.shape[0]}")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        X = df.drop(columns=[self.target], axis=1)
        y = df[self.target]
        
        stratify = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=stratify
        )
        
        logger.info(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
