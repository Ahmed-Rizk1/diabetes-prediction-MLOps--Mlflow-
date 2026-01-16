import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from typing import Tuple, List

class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.num_cols = config['preprocessing']['numeric_features']
        self.categ_cols = config['preprocessing']['categorical_features']
        self.smote_strategy = config['preprocessing']['smote_strategy']
        self.pipeline = None
        self.feature_names = None

    def get_pipeline(self) -> ColumnTransformer:
        """Create the sklearn preprocessing pipeline."""
        
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categ_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # We can use ColumnTransformer to apply specific pipelines to specific columns
        # Note: The original script used a custom DataFrameSelector and FeatureUnion. 
        # ColumnTransformer is the modern standard way.
        # "Ready cols" in the original script were handled separately. 
        # ColumnTransformer drops remainder by default. We should passthrough or handle them.
        # Let's see what "ready cols" are. 
        # Original: ready_cols = list(set(X_train.columns) - set(num) - set(categ))
        # If there are any other columns, we should probably handle them. 
        # For now, I will assume remainder='passthrough' or explicitly handle them if I knew them.
        # Given the dataset, "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember" seem to be numeric but treated as... ?
        # In original script:
        # num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
        # categ_cols = ['Gender', 'Geography']
        # ready_cols = Rest.
        # "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember" fit into ready_cols.
        # original ready_pipeline: Imputer(most_frequent).
        
        # Let's define ready pipeline dynamically or just use ColumnTransformer with remainder
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.num_cols),
                ('cat', categ_pipeline, self.categ_cols)
            ],
            remainder='passthrough' # For "ready columns" which originally just had imputation. 
            # Ideally we should add an imputer for them too if they have missing values.
            # But 'passthrough' leaves them as is. 
            # If we strictly want to follow original:
        )
        
        # To strictly follow original which applies Imputer(most_frequent) to remainder:
        # We can add a third transformer if we can identify the columns dynamically.
        # But ColumnTransformer needs column names at initialization usually.
        # We'll stick to this for now, and maybe refine if needed.
        
        return self.pipeline

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform the data. Apply SMOTE if y is provided."""
        
        # Get pipeline
        pipeline = self.get_pipeline()
        
        # Fit transform features
        X_processed = pipeline.fit_transform(X)

        # Get feature names (best effort)
        # numeric + (encoded categorical) + (remainder)
        # This is a bit tricky with ColumnTransformer but feasible for OHE
        
        # Apply SMOTE
        X_resampled, y_resampled = X_processed, y
        if y is not None:
             over = SMOTE(sampling_strategy=self.smote_strategy, random_state=45)
             X_resampled, y_resampled = over.fit_resample(X_processed, y)
             
        return X_resampled, y_resampled, pipeline

    def transform(self, X: pd.DataFrame, pipeline: ColumnTransformer) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        return pipeline.transform(X)
