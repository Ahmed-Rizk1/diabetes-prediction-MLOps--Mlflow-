import argparse
import sys
import os
import logging

# Add src to python path if needed, though running from root usually works if __init__ present
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.pipelines.training import run_training_pipeline
from src.pipelines.inference import run_inference_pipeline
from src.config.loader import load_config
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Bank Churn Prediction Pipeline")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--n_estimators", type=int, help="Number of trees")
    train_parser.add_argument("--max_depth", type=int, help="Max depth of trees")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on data")
    predict_parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV")
    predict_parser.add_argument("--output_file", type=str, help="Path to save predictions")
    
    args = parser.parse_args()
    
    if args.command == "train":
        logger.info("Starting training pipeline...")
        run_training_pipeline(args)
        
    elif args.command == "predict":
        logger.info("Starting inference...")
        run_inference_pipeline(args.input_file, args.output_file)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
