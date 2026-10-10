"""Model definitions, training loops, and validation metrics."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Instantiate classifier based on configuration parameters."""
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(**params)
    elif model_name == "svm":
        return SVC(**params)
    else:
        raise ValueError(f"Unsupported model identifier: {model_name}")


def evaluate_cross_validation(
    model: Any, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5
) -> Dict[str, float]:
    """Perform Stratified K-Fold cross validation and record metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="weighted"))

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_weighted_f1": float(np.mean(f1s)),
    }
