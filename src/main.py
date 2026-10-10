"""CLI entrypoint for multivariate fault detection model training."""

import argparse
from src.utils.data_loader import load_config, prepare_dataset, scale_features
from src.models.train import get_model, evaluate_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial Fault Classification Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--data-path", type=str, required=True, help="Path to telemetry dataset CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    target_col = config["data"]["target_column"]

    X_train, X_test, y_train, y_test, label_encoder = prepare_dataset(
        args.data_path, target_col=target_col, test_size=config["data"]["test_size"]
    )
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    print(f"Loaded dataset with {len(label_encoder.classes_)} fault classes: {list(label_encoder.classes_)}")
    print(f"Training samples: {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")

    for model_name, params in config["models"].items():
        print(f"\n--- Evaluating {model_name.upper()} ---")
        model = get_model(model_name, params)
        metrics = evaluate_cross_validation(model, X_train_scaled, y_train, n_splits=config["data"]["cv_folds"])
        print(f"Stratified CV Accuracy: {metrics['mean_accuracy']:.4f} (±{metrics['std_accuracy']:.4f})")
        print(f"Stratified CV Weighted F1: {metrics['mean_weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
