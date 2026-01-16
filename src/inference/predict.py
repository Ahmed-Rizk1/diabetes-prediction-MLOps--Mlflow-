import pandas as pd
import joblib
import os
from src.config.loader import load_config
import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config: dict):
        self.config = config
        self.artifacts_dir = config['paths']['artifacts']
        self.pipeline_path = os.path.join(self.artifacts_dir, 'preprocessing_pipeline.joblib')
        self.model_path = os.path.join(self.artifacts_dir, 'model.joblib')
        self.pipeline = None
        self.model = None

    def load_artifacts(self):
        """Load model and pipeline."""
        if not os.path.exists(self.pipeline_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Artifacts not found in {self.artifacts_dir}. Run training first.")
        
        logger.info("Loading artifacts...")
        self.pipeline = joblib.load(self.pipeline_path)
        self.model = joblib.load(self.model_path)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with predictions attached.
        """
        if self.model is None:
            self.load_artifacts()
            
        # Apply preprocessing
        # Note: The pipeline expects raw data structure (columns) as seen during training.
        try:
            data_processed = self.pipeline.transform(data)
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise e
            
        # Predict
        predictions = self.model.predict(data_processed)
        probabilities = self.model.predict_proba(data_processed)[:, 1]
        
        # Return results
        results = data.copy()
        results['prediction'] = predictions
        results['churn_probability'] = probabilities
        
        return results

if __name__ == "__main__":
    # Example usage
    config = load_config()
    predictor = Predictor(config)
    # predictor.predict(some_df)
