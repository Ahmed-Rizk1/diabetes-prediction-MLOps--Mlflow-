from src.config.loader import load_config
from src.inference.predict import Predictor
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def run_inference_pipeline(input_path: str, output_path: str = None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    config = load_config()
    predictor = Predictor(config)
    
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    logger.info("Running inference...")
    results = predictor.predict(df)
    
    if output_path:
        logger.info(f"Saving predictions to {output_path}...")
        results.to_csv(output_path, index=False)
    else:
        print(results[['prediction', 'churn_probability']].head())
    
    return results
