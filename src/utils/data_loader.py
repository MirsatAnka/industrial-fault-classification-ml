"""Data loading, scaling, and configuration parsing utilities."""

from typing import Dict, Any, Tuple
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_config(config_path: str) -> Dict[str, Any]:
    """Parse YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    filepath: str, target_col: str, test_size: float = 0.2, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder]:
    """Load, encode labels, and stratify train/test splits."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_col])
    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    y_raw = df[target_col]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, label_encoder


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train set and apply to test set to avoid leakage."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
