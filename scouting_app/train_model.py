"""Entrenamiento del modelo de prediccion de potencial."""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
from preprocessing import (
    build_preprocessor,
    fit_transform_features,
    preprocessor_input_dim,
    save_preprocessor,
    training_dataframe_from_engine,
    transform_features,
)

SEED = int(os.environ.get("SEED", "42"))
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_data(db_url: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Carga los jugadores y devuelve DataFrame de features + etiquetas."""
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    df = training_dataframe_from_engine(engine)
    if df.empty:
        raise ValueError("No hay jugadores disponibles para entrenar el modelo.")
    y = df["potential_label"].astype(np.float32).to_numpy()
    return df, y


def train_model(features_df: pd.DataFrame, y: np.ndarray, epochs: int = 10, lr: float = 1e-3):
    """Entrena la red neuronal y devuelve el modelo junto al preprocesador."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    stratify = y if len(unique_classes) > 1 and class_counts.min() >= 2 else None
    try:
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            features_df,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is None:
            raise
        print(f"Advertencia: no se pudo estratificar el split: {exc}. Se continua sin estratificacion.")
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            features_df,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=None,
        )

    preprocessor = build_preprocessor()
    X_train, preprocessor = fit_transform_features(X_train_df, preprocessor)
    X_test = transform_features(X_test_df, preprocessor)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = PlayerNet(input_dim=preprocessor_input_dim(preprocessor))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Epoca {epoch + 1}/{epochs}, perdida: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        preds_binary = (preds >= 0.5).float()
        accuracy = (preds_binary.eq(y_test_tensor)).float().mean().item()
    print(f"Precision en el conjunto de prueba: {accuracy * 100:.2f}%")

    y_true = y_test_tensor.cpu().numpy().reshape(-1)
    y_prob = preds.cpu().numpy().reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para ROC-AUC.")
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception as exc:
        roc_auc = None
        print(f"Advertencia: no se pudo calcular ROC-AUC: {exc}")

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para PR-AUC.")
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception as exc:
        pr_auc = None
        print(f"Advertencia: no se pudo calcular PR-AUC: {exc}")

    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception as exc:
        f1 = precision = recall = None
        cm = None
        print(f"Advertencia: no se pudieron calcular metricas de clasificacion: {exc}")

    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC: {pr_auc:.4f}")
    if f1 is not None:
        print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    if cm is not None:
        print("Confusion matrix:", cm)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else "",
        "pr_auc": float(pr_auc) if pr_auc is not None else "",
        "f1": float(f1) if f1 is not None else "",
        "precision": float(precision) if precision is not None else "",
        "recall": float(recall) if recall is not None else "",
        "train_size": int(len(X_train_df)),
        "test_size": int(len(X_test_df)),
    }
    return model, preprocessor, metrics


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def main(db_url: str, model_out: str, preprocessor_out: str, epochs: int, lr: float) -> None:
    feature_df, y = load_data(db_url)
    model, preprocessor, metrics = train_model(feature_df, y, epochs=epochs, lr=lr)
    save_model(model, model_out)
    save_preprocessor(preprocessor, preprocessor_out)
    print(f"Modelo guardado en {model_out}")
    print(f"Preprocesador guardado en {preprocessor_out}")

    try:
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "seed": SEED,
            "epochs": int(epochs),
            "lr": float(lr),
            "model_path": model_out,
            "preprocessor_path": preprocessor_out,
            **metrics,
        }
        log_experiment(row)
    except Exception as exc:
        print(f"Advertencia: no se pudo registrar la corrida del experimento: {exc}")


def log_experiment(row: dict) -> None:
    exp_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    write_header = not os.path.exists(exp_path)
    with open(exp_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    args = parser.parse_args()
    main(args.db_url, args.model_out, args.preprocessor_out, args.epochs, args.lr)
