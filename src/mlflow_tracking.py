"""Shared MLflow tracking for the churn prediction project."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
EXPERIMENT_NAME = "churn-prediction"


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    """Point MLflow at the project SQLite store and select an experiment."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=ARTIFACT_ROOT.resolve().as_uri(),
        )
    mlflow.set_experiment(experiment_name)


def enable_autolog() -> None:
    """Capture sklearn / XGBoost params, metrics, and models automatically."""
    for module_name, autolog_name in (("sklearn", "autolog"), ("xgboost", "autolog")):
        module = getattr(mlflow, module_name, None)
        if module is None or not hasattr(module, autolog_name):
            continue
        try:
            getattr(module, autolog_name)(log_models=True, silent=True)
        except TypeError:
            getattr(module, autolog_name)(log_models=True)


def _positive_class_probabilities(y_prob):
    """Return a 1D array containing only the positive-class probabilities."""
    values = np.asarray(y_prob)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    if values.shape[1] == 2:
        return values[:, 1]
    if values.shape[1] == 1:
        return values[:, 0]
    return values.reshape(-1)


def log_params_safe(params: dict) -> None:
    cleaned = {k: ("None" if v is None else v) for k, v in params.items()}
    mlflow.log_params(cleaned)


def log_dataset_info(X, y=None, prefix: str = "data") -> None:
    n_rows = getattr(X, "shape", [len(X)])[0]
    n_cols = getattr(X, "shape", [None, None])[1] if hasattr(X, "shape") and len(X.shape) > 1 else None
    mlflow.log_metric(f"{prefix}_n_rows", int(n_rows))
    if n_cols is not None:
        mlflow.log_metric(f"{prefix}_n_features", int(n_cols))
    if y is not None:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        for value, count in zip(values, counts):
            mlflow.log_metric(f"{prefix}_class_{value}_count", int(count))


def log_classification_metrics(y_true, y_pred, y_prob=None, prefix: str = "") -> dict:
    key = lambda name: f"{prefix}{name}" if prefix else name
    metrics = {
        key("accuracy"): float(accuracy_score(y_true, y_pred)),
        key("precision"): float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        key("recall"): float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        key("f1"): float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_prob is not None:
        positive_probs = _positive_class_probabilities(y_prob)
        metrics[key("roc_auc")] = float(roc_auc_score(y_true, positive_probs))
    mlflow.log_metrics(metrics)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, int(value), ha="center", va="center")
        fig.tight_layout()
        cm_path = Path(tmp) / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))
    return metrics


def log_model(model, name: str, X_sample=None) -> None:
    flavor = "xgboost" if model.__class__.__name__.lower().startswith("xgb") else "sklearn"
    extra = {}
    if X_sample is not None:
        try:
            from mlflow.models import infer_signature

            sample = X_sample[:5]
            extra["signature"] = infer_signature(sample, model.predict(sample))
        except Exception:
            pass

    module = getattr(mlflow, flavor, None)
    if module is None or not hasattr(module, "log_model"):
        raise AttributeError(f"MLflow does not provide a '{flavor}' model logger")

    logger = module.log_model
    try:
        logger(model, name=name, **extra)
    except TypeError:
        logger(model, artifact_path=name, **extra)


def main() -> None:
    """Initialize the project MLflow experiment and print the configured tracking URI."""
    setup_mlflow()
    print(f"MLflow tracking URI: {TRACKING_URI}")


if __name__ == "__main__":
    main()
