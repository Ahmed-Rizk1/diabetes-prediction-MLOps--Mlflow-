# Bank Customer Churn Prediction

## Overview
This project implements a production-ready Machine Learning pipeline to predict bank customer churn. It demonstrates professional software engineering practices, modular architecture, and MLOps principles using **MLflow** for experiment tracking.

## Architecture
The project is organized into a modular structure to ensure scalability and maintainability:

```
├── config/             # Configuration files
├── data/               # Data storage (raw & processed)
├── src/                # Source code
│   ├── config/         # Config loading
│   ├── data/           # Data loading & cleaning
│   ├── features/       # Feature engineering & preprocessing
│   ├── models/         # Model training
│   ├── evaluation/     # Evaluation metrics & plotting
│   ├── inference/      # Inference logic
│   └── pipelines/      # End-to-end pipelines
├── tests/              # Unit tests
├── notebooks/          # Exploration notebooks
├── scripts/            # Helper scripts
├── main.py             # CLI entry point
└── requirements.txt    # Dependencies
```

## Tech Stack
- **Languages**: Python 3.9+
- **ML Libraries**: Scikit-Learn, Imbalanced-Learn (SMOTE), Pandas, NumPy
- **MLOps**: MLflow
- **Testing**: Pytest

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
Run the training pipeline. This will load data, preprocess it, train a Random Forest model, and log results to MLflow.
```bash
python main.py train --n_estimators 100 --max_depth 10
```

### Inference
Generate predictions on a new dataset.
```bash
python main.py predict --input_file data/raw/dataset.csv --output_file predictions.csv
```

### Testing
Run unit tests to verify components.
```bash
pytest tests/
```

## Results
- **Metrics**: The model evaluates Accuracy and F1-Score.
- **Artifacts**: Confusion Matrix and ROC Curve are logged to MLflow and saved in `reports/figures/`.

## Future Improvements
- [ ] Add Docker support for containerization.
- [ ] Deploy as a FastAPI service.
- [ ] Integrate DVC for data versioning.