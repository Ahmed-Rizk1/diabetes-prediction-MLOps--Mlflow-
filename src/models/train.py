import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Any

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.n_estimators = config['training']['n_estimators']
        self.max_depth = config['training']['max_depth']
        self.random_state = config['training']['random_state']
        self.experiment_name = config['mlflow']['experiment_name']
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the model."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run() as run:
            mlflow.set_tag('clf', 'forest')
            
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            
            print(f"Training RandomForest with n_estimators={self.n_estimators}, max_depth={self.max_depth}...")
            self.model.fit(X_train, y_train)
            
            # Log params
            mlflow.log_params({
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth
            })
            
            # Log model
            mlflow.sklearn.log_model(self.model, "random_forest_model")
            
        return self.model

    def evaluate(self, model, X_test, y_test):
        """Evaluate the model and log metrics."""
        # Note: mlflow run might be closed, so we might need to be inside a run or accept a run_id
        # Ideally evaluation happens inside the same run or a new run.
        # For simplicity, we assume this is called immediately or we start a new run context if needed.
        # But usually we want metrics in the SAME run as training.
        
        # Let's refactor: train_and_evaluate usually is better for script based flow.
        pass 
