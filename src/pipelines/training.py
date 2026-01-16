from src.config.loader import load_config
from src.data.loader import DataLoader
from src.features.preprocessing import Preprocessor
from src.models.train import ModelTrainer
from src.evaluation.evaluate import Evaluator
import joblib
import os
import argparse

def run_training_pipeline(args):
    # Load config
    config = load_config()
    
    # 1. Load Data
    loader = DataLoader(config)
    df = loader.load_data()
    df = loader.clean_data(df)
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    # 2. Preprocess
    preprocessor = Preprocessor(config)
    # Fit transform on train
    X_train_processed, y_train_resampled, pipeline = preprocessor.fit_transform(X_train, y_train)
    # Transform test
    X_test_processed = preprocessor.transform(X_test, pipeline)
    
    # Save Pipeline
    os.makedirs(config['paths']['artifacts'], exist_ok=True)
    joblib.dump(pipeline, os.path.join(config['paths']['artifacts'], 'preprocessing_pipeline.joblib'))
    
    # 3. Model Training
    trainer = ModelTrainer(config)
    # Override config with args if provided
    if args.n_estimators:
        trainer.n_estimators = args.n_estimators
    if args.max_depth:
        trainer.max_depth = args.max_depth
        
    model = trainer.train(X_train_processed, y_train_resampled)
    
    # Save Model
    joblib.dump(model, os.path.join(config['paths']['artifacts'], 'model.joblib'))
    
    # 4. Evaluation
    evaluator = Evaluator(config)
    evaluator.evaluate(model, X_test_processed, y_test, prefix="rf_model")
    
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, help="Number of trees")
    parser.add_argument("--max_depth", type=int, help="Max depth of trees")
    args = parser.parse_args()
    
    run_training_pipeline(args)
