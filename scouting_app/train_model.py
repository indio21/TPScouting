"""Entrenamiento del modelo de prediccion de potencial."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import Base
from player_logic import ATTRIBUTE_FIELDS, EVAL_MAX_AGE, EVAL_MIN_AGE
from preprocessing import (
    build_preprocessor,
    fit_transform_features,
    preprocessor_input_dim,
    save_preprocessor,
    TEMPORAL_TARGET_COLUMN,
    temporal_training_dataframe_from_engine,
    training_dataframe_from_engine,
    transform_features,
)

SEED = int(os.environ.get("SEED", "42"))
DEFAULT_THRESHOLDS = np.linspace(0.10, 0.90, 33)
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.17647058823529413  # 15% del total tras separar test

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PlayerNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_data(
    db_url: str,
    age_min: int = EVAL_MIN_AGE,
    age_max: int = EVAL_MAX_AGE,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """Carga jugadores y devuelve features, etiquetas y resumen del dataset."""
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    raw_df = temporal_training_dataframe_from_engine(engine)
    if raw_df.empty:
        raise ValueError("No hay jugadores disponibles para entrenar el modelo.")

    filtered_df = raw_df[(raw_df["age"] >= age_min) & (raw_df["age"] <= age_max)].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No hay jugadores disponibles en el rango etario {age_min}-{age_max} para entrenar el modelo."
        )

    y = filtered_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy()
    summary = dataset_summary(raw_df, filtered_df, TEMPORAL_TARGET_COLUMN)
    return filtered_df, y, summary


def dataset_summary(raw_df: pd.DataFrame, filtered_df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    y_filtered = filtered_df[target_column].astype(int)
    positive_count = int(y_filtered.sum())
    total_count = int(len(filtered_df))
    negative_count = int(total_count - positive_count)
    return {
        "raw_rows": int(len(raw_df)),
        "filtered_rows": total_count,
        "age_range_raw": {
            "min": int(raw_df["age"].min()),
            "max": int(raw_df["age"].max()),
        },
        "age_range_filtered": {
            "min": int(filtered_df["age"].min()),
            "max": int(filtered_df["age"].max()),
        },
        "class_distribution": {
            "positive": positive_count,
            "negative": negative_count,
            "positive_rate": round(positive_count / total_count, 4) if total_count else 0.0,
        },
        "position_distribution": {
            str(key): int(value)
            for key, value in filtered_df["position"].value_counts().sort_index().items()
        },
        "target_column": target_column,
    }


def choose_stratify_target(y: np.ndarray) -> Optional[np.ndarray]:
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) <= 1 or class_counts.min() < 2:
        return None
    return y


def safe_train_test_split(*arrays, test_size: float, stratify: Optional[np.ndarray], random_state: int):
    try:
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is None:
            raise
        print(f"Advertencia: no se pudo estratificar el split: {exc}. Se continua sin estratificacion.")
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": "",
            "roc_auc": "",
            "pr_auc": "",
            "f1": "",
            "precision": "",
            "recall": "",
            "confusion_matrix": [],
        }

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float((y_pred == y_true).mean())

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para ROC-AUC.")
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = ""

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para PR-AUC.")
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = ""

    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        f1 = precision = recall = ""
        cm = []

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, object]]:
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, y_prob, best_threshold)
    best_f1 = float(best_metrics["f1"] or 0.0)
    best_recall = float(best_metrics["recall"] or 0.0)

    for threshold in DEFAULT_THRESHOLDS:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        f1 = float(metrics["f1"] or 0.0)
        recall = float(metrics["recall"] or 0.0)
        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = f1
            best_recall = recall
    return best_threshold, best_metrics


def tensor_from_array(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def avg_skill_scores(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=np.float32)
    return (df.loc[:, ATTRIBUTE_FIELDS].mean(axis=1).to_numpy(dtype=np.float32) / 20.0).reshape(-1)


def train_model(
    features_df: pd.DataFrame,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 8,
):
    """Entrena la red y devuelve modelo, preprocesador y metadata completa."""
    splits = safe_train_test_split(
        features_df,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=SEED,
        stratify=choose_stratify_target(y),
    )
    X_train_val_df, X_test_df, y_train_val, y_test = splits
    splits = safe_train_test_split(
        X_train_val_df,
        y_train_val,
        test_size=DEFAULT_VAL_SIZE,
        random_state=SEED,
        stratify=choose_stratify_target(y_train_val),
    )
    X_train_df, X_val_df, y_train, y_val = splits

    preprocessor = build_preprocessor()
    X_train, preprocessor = fit_transform_features(X_train_df, preprocessor)
    X_val = transform_features(X_val_df, preprocessor)
    X_test = transform_features(X_test_df, preprocessor)

    X_train_tensor = tensor_from_array(X_train)
    y_train_tensor = tensor_from_array(y_train).view(-1, 1)
    X_val_tensor = tensor_from_array(X_val)
    X_test_tensor = tensor_from_array(X_test)

    positive_count = float(np.sum(y_train))
    negative_count = float(len(y_train) - positive_count)
    pos_weight_value = negative_count / positive_count if positive_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    model = PlayerNet(input_dim=preprocessor_input_dim(preprocessor))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_threshold = 0.5
    best_epoch = 1
    best_monitor = (-1.0, -1.0)
    best_val_metrics: Dict[str, object] = classification_metrics(y_val, np.zeros(len(y_val)), 0.5)
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor).cpu().numpy().reshape(-1)
        val_prob = sigmoid_numpy(val_logits)
        threshold, val_metrics = select_best_threshold(y_val, val_prob)
        monitor_pr_auc = float(val_metrics["pr_auc"] or 0.0)
        monitor_f1 = float(val_metrics["f1"] or 0.0)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "val_pr_auc": monitor_pr_auc,
                "val_f1": monitor_f1,
                "threshold": float(threshold),
            }
        )

        is_better = (monitor_pr_auc, monitor_f1) > best_monitor
        if is_better:
            best_monitor = (monitor_pr_auc, monitor_f1)
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoca {epoch + 1}/{epochs}, perdida: {loss.item():.4f}, "
                f"val_pr_auc: {monitor_pr_auc:.4f}, val_f1: {monitor_f1:.4f}, threshold: {threshold:.2f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping activado en epoca {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_tensor).cpu().numpy().reshape(-1)
    test_prob = sigmoid_numpy(test_logits)
    test_metrics = classification_metrics(y_test, test_prob, best_threshold)

    baseline_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    baseline_model.fit(X_train, y_train)
    baseline_val_prob = baseline_model.predict_proba(X_val)[:, 1]
    baseline_threshold, baseline_val_metrics = select_best_threshold(y_val, baseline_val_prob)
    baseline_test_prob = baseline_model.predict_proba(X_test)[:, 1]
    baseline_test_metrics = classification_metrics(y_test, baseline_test_prob, baseline_threshold)

    avg_skill_val_prob = avg_skill_scores(X_val_df)
    avg_skill_threshold, avg_skill_val_metrics = select_best_threshold(y_val, avg_skill_val_prob)
    avg_skill_test_prob = avg_skill_scores(X_test_df)
    avg_skill_test_metrics = classification_metrics(y_test, avg_skill_test_prob, avg_skill_threshold)

    print(f"Precision en test (PyTorch): {float(test_metrics['accuracy']) * 100:.2f}%")
    if test_metrics["roc_auc"] != "":
        print(f"ROC-AUC (PyTorch): {float(test_metrics['roc_auc']):.4f}")
    if test_metrics["pr_auc"] != "":
        print(f"PR-AUC (PyTorch): {float(test_metrics['pr_auc']):.4f}")
    if test_metrics["f1"] != "":
        print(
            "F1 (PyTorch): "
            f"{float(test_metrics['f1']):.4f} | "
            f"Precision: {float(test_metrics['precision']):.4f} | "
            f"Recall: {float(test_metrics['recall']):.4f}"
        )
    print("Confusion matrix (PyTorch):", test_metrics["confusion_matrix"])
    print(
        "Baseline LogisticRegression balanced | "
        f"ROC-AUC: {baseline_test_metrics['roc_auc']} | "
        f"PR-AUC: {baseline_test_metrics['pr_auc']} | "
        f"F1: {baseline_test_metrics['f1']}"
    )
    print(
        "Baseline promedio atributos | "
        f"ROC-AUC: {avg_skill_test_metrics['roc_auc']} | "
        f"PR-AUC: {avg_skill_test_metrics['pr_auc']} | "
        f"F1: {avg_skill_test_metrics['f1']}"
    )

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "seed": SEED,
        "config": {
            "epochs_requested": int(epochs),
            "epochs_trained": int(history[-1]["epoch"]) if history else 0,
            "learning_rate": float(lr),
            "patience": int(patience),
            "age_min": int(EVAL_MIN_AGE),
            "age_max": int(EVAL_MAX_AGE),
            "loss": "BCEWithLogitsLoss",
            "target_type": "temporal_progression",
            "pos_weight": float(pos_weight_value),
            "test_size": float(DEFAULT_TEST_SIZE),
            "validation_size_on_train_pool": float(DEFAULT_VAL_SIZE),
        },
        "dataset": {
            "train_size": int(len(X_train_df)),
            "validation_size": int(len(X_val_df)),
            "test_size": int(len(X_test_df)),
            "train_positive_rate": round(float(np.mean(y_train)), 4),
            "validation_positive_rate": round(float(np.mean(y_val)), 4),
            "test_positive_rate": round(float(np.mean(y_test)), 4),
        },
        "pytorch": {
            "best_epoch": int(best_epoch),
            "selected_threshold": float(best_threshold),
            "validation": best_val_metrics,
            "test": test_metrics,
            "history": history,
        },
        "baselines": {
            "logistic_regression_balanced": {
                "selected_threshold": float(baseline_threshold),
                "validation": baseline_val_metrics,
                "test": baseline_test_metrics,
            },
            "avg_skill_score": {
                "selected_threshold": float(avg_skill_threshold),
                "validation": avg_skill_val_metrics,
                "test": avg_skill_test_metrics,
            },
        },
    }
    return model, preprocessor, metadata


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_metadata(metadata: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main(
    db_url: str,
    model_out: str,
    preprocessor_out: str,
    metadata_out: str,
    epochs: int,
    lr: float,
    patience: int = 8,
) -> None:
    feature_df, y, data_summary = load_data(db_url)
    model, preprocessor, metadata = train_model(feature_df, y, epochs=epochs, lr=lr, patience=patience)
    metadata["dataset_summary"] = data_summary
    save_model(model, model_out)
    save_preprocessor(preprocessor, preprocessor_out)
    save_metadata(metadata, metadata_out)
    print(f"Modelo guardado en {model_out}")
    print(f"Preprocesador guardado en {preprocessor_out}")
    print(f"Metadata guardada en {metadata_out}")

    try:
        row = {
            "timestamp": metadata["timestamp"],
            "seed": SEED,
            "epochs": int(metadata["config"]["epochs_trained"]),
            "lr": float(lr),
            "model_path": model_out,
            "preprocessor_path": preprocessor_out,
            "accuracy": metadata["pytorch"]["test"]["accuracy"],
            "roc_auc": metadata["pytorch"]["test"]["roc_auc"],
            "pr_auc": metadata["pytorch"]["test"]["pr_auc"],
            "f1": metadata["pytorch"]["test"]["f1"],
            "precision": metadata["pytorch"]["test"]["precision"],
            "recall": metadata["pytorch"]["test"]["recall"],
            "train_size": metadata["dataset"]["train_size"],
            "test_size": metadata["dataset"]["test_size"],
        }
        log_experiment(row)
    except Exception as exc:
        print(f"Advertencia: no se pudo registrar la corrida del experimento: {exc}")


def log_experiment(row: dict) -> None:
    exp_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    expected_fields = [
        "timestamp",
        "seed",
        "epochs",
        "lr",
        "model_path",
        "preprocessor_path",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "train_size",
        "test_size",
    ]
    write_header = not os.path.exists(exp_path)
    with open(exp_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=expected_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in expected_fields})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de scouting")
    parser.add_argument(
        "--db-url",
        type=str,
        default="sqlite:///players_training.db",
        help="URL de la base de datos de entrenamiento",
    )
    parser.add_argument("--model-out", type=str, default="model.pt", help="Ruta de salida del modelo")
    parser.add_argument(
        "--preprocessor-out",
        type=str,
        default="preprocessor.joblib",
        help="Ruta de salida del preprocesador",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default="training_metadata.json",
        help="Ruta de salida de metadata del entrenamiento",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Numero de epocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--patience", type=int, default=8, help="Paciencia para early stopping")
    args = parser.parse_args()
    main(args.db_url, args.model_out, args.preprocessor_out, args.metadata_out, args.epochs, args.lr, args.patience)
