"""Preprocesamiento compartido para entrenamiento e inferencia."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sqlalchemy import select

from models import Player, PlayerStat
from player_logic import ATTRIBUTE_FIELDS, POSITION_CHOICES

HISTORICAL_FEATURE_COLUMNS: List[str] = [
    "stats_entry_count",
    "avg_final_score_hist",
    "avg_pass_accuracy_hist",
    "latest_final_score_hist",
]
NUMERIC_FEATURE_COLUMNS: List[str] = ["age", *ATTRIBUTE_FIELDS, *HISTORICAL_FEATURE_COLUMNS]
CATEGORICAL_FEATURE_COLUMNS: List[str] = ["position"]
MODEL_FEATURE_COLUMNS: List[str] = [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    categories=[POSITION_CHOICES],
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def training_dataframe_from_engine(engine) -> pd.DataFrame:
    query = select(
        Player.id.label("player_id"),
        Player.age,
        Player.pace,
        Player.shooting,
        Player.passing,
        Player.dribbling,
        Player.defending,
        Player.physical,
        Player.vision,
        Player.tackling,
        Player.determination,
        Player.technique,
        Player.position,
        Player.potential_label,
    )
    stats_query = select(
        PlayerStat.player_id,
        PlayerStat.record_date,
        PlayerStat.pass_accuracy,
        PlayerStat.final_score,
    )
    with engine.connect() as connection:
        players_df = pd.read_sql(query, connection)
        stats_df = pd.read_sql(stats_query, connection)
    return merge_historical_features(players_df, stats_df)


def dataframe_from_players(
    players: Iterable[Player],
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> pd.DataFrame:
    rows = []
    stats_feature_map = stats_feature_map or {}
    for player in players:
        stats_features = stats_feature_map.get(player.id, {})
        rows.append(
            {
                "player_id": player.id,
                "age": player.age,
                "pace": player.pace,
                "shooting": player.shooting,
                "passing": player.passing,
                "dribbling": player.dribbling,
                "defending": player.defending,
                "physical": player.physical,
                "vision": player.vision,
                "tackling": player.tackling,
                "determination": player.determination,
                "technique": player.technique,
                "position": player.position,
                **historical_feature_defaults(stats_features),
            }
        )
    return pd.DataFrame(rows, columns=["player_id", *MODEL_FEATURE_COLUMNS])


def feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, MODEL_FEATURE_COLUMNS].copy()


def fit_transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer | None = None):
    transformer = preprocessor or build_preprocessor()
    transformed = transformer.fit_transform(feature_dataframe(df))
    return to_float32_array(transformed), transformer


def transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    transformed = preprocessor.transform(feature_dataframe(df))
    return to_float32_array(transformed)


def to_float32_array(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def preprocessor_input_dim(preprocessor: ColumnTransformer) -> int:
    return len(preprocessor.get_feature_names_out())


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    return load(path)


def historical_feature_defaults(values: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Optional[float]]:
    source = values or {}
    return {
        "stats_entry_count": source.get("stats_entry_count"),
        "avg_final_score_hist": source.get("avg_final_score_hist"),
        "avg_pass_accuracy_hist": source.get("avg_pass_accuracy_hist"),
        "latest_final_score_hist": source.get("latest_final_score_hist"),
    }


def aggregate_stats_dataframe(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame(columns=["player_id", *HISTORICAL_FEATURE_COLUMNS])

    working_df = stats_df.copy()
    working_df["record_date"] = pd.to_datetime(working_df["record_date"], errors="coerce")
    grouped = (
        working_df.groupby("player_id", as_index=False)
        .agg(
            stats_entry_count=("player_id", "size"),
            avg_final_score_hist=("final_score", "mean"),
            avg_pass_accuracy_hist=("pass_accuracy", "mean"),
        )
    )
    latest_df = (
        working_df.sort_values(["player_id", "record_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "final_score"]]
        .rename(columns={"final_score": "latest_final_score_hist"})
    )
    merged = grouped.merge(latest_df, on="player_id", how="left")
    return merged


def merge_historical_features(players_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    historical_df = aggregate_stats_dataframe(stats_df)
    merged = players_df.merge(historical_df, on="player_id", how="left")
    for column in HISTORICAL_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged
