import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import mlflow
import os

class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.plots_dir = config['paths']['plots']
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate(self, model, X_test, y_test, prefix="model"):
        """Evaluate model and log metrics/plots to MLflow."""
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[{prefix}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        mlflow.log_metrics({
            f"{prefix}_accuracy": acc,
            f"{prefix}_f1_score": f1
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, prefix)
        
        # ROC Curve
        # Check if model has predict_proba
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            self.plot_roc_curve(y_test, y_probs, prefix)

        return acc, f1

    def plot_confusion_matrix(self, cm, prefix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cbar=False, fmt='d', cmap='Blues')
        plt.title(f'{prefix} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_path = os.path.join(self.plots_dir, f"{prefix}_conf_matrix.png")
        plt.savefig(plot_path)
        mlflow.log_figure(plt.gcf(), f"{prefix}_conf_matrix.png")
        plt.close()

    def plot_roc_curve(self, y_test, y_probs, prefix):
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{prefix} ROC Curve')
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(self.plots_dir, f"{prefix}_roc_curve.png")
        plt.savefig(plot_path)
        mlflow.log_figure(plt.gcf(), f"{prefix}_roc_curve.png")
        plt.close()
